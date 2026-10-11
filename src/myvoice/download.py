"""``./myvoice.py download``: fetch the model files into ./models.

Files that are already there are skipped, so running it again is cheap. Every file is pinned to an
explicit repo + file name (see to-download.md). Downloads are plain HTTPS and resumable: partial
data stays in FILE.part, and the next attempt (or the next run) continues from there. Finished
files are checked against their size and sha256.

Not ``hf download``: its Xet transfer aborts on slow/unstable lines ("CAS Client Error ... error
decoding response body") and throws the partial data away.
"""

from __future__ import annotations

import hashlib
import http.client
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

from .config import PROJECT_ROOT

MODELS = PROJECT_ROOT / "models"
HUB = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
HEADERS = {"User-Agent": "myvoice-download/1.0"}
CHUNK = 1 << 20
TIMEOUT_S = 60  # a connection that delivers nothing for a minute counts as dropped
MAX_STALLS = 10  # attempts in a row without progress before giving up (until the next run)
RETRY_S = 10

WHISPER_FILES = ("config.json", "preprocessor_config.json", "tokenizer.json", "vocabulary.json", "model.bin")
LLMS = {  # name: (repo, file, prompt_style)
    "translategemma-q6": ("mradermacher/translategemma-4b-it-GGUF", "translategemma-4b-it.Q6_K.gguf", "translategemma"),
    "translategemma-q4": ("mradermacher/translategemma-4b-it-GGUF", "translategemma-4b-it.Q4_K_M.gguf", "translategemma"),
    "translategemma-q8": ("mradermacher/translategemma-4b-it-GGUF", "translategemma-4b-it.Q8_0.gguf", "translategemma"),
    "qwen3.5-4b": ("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q5_K_M.gguf", "chat"),
    "gemma4-e4b": ("ggml-org/gemma-4-E4B-it-GGUF", "gemma-4-E4B-it-Q4_0.gguf", "chat"),
}
DEFAULT_LLM = "translategemma-q6"
PIPER_VOICES = (
    "it/it_IT/paola/medium/it_IT-paola-medium.onnx",  # Italian, female
    "it/it_IT/serena/high/it_IT-serena-high.onnx",  # Italian, female
    "it/it_IT/riccardo/x_low/it_IT-riccardo-x_low.onnx",  # Italian, male (low quality)
    "en/en_US/lessac/high/en_US-lessac-high.onnx",  # US English, female
    "en/en_US/ryan/high/en_US-ryan-high.onnx",  # US English, male
    "en/en_GB/cori/high/en_GB-cori-high.onnx",  # British English, female
    "en/en_GB/alan/medium/en_GB-alan-medium.onnx",  # British English, male
)
# Kokoro-82M (Apache-2.0), ONNX export from the kokoro-onnx project; sha256 from its release page
KOKORO_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
KOKORO_FILES = {
    "kokoro-v1.0.onnx": "beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}
GROUPS = ("whisper", "llm", "voices", "kokoro")


class DownloadError(RuntimeError):
    pass


@dataclass
class Remote:
    size: int
    sha256: str | None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: ANN002, ANN003 - urllib signature
        return None  # 3xx comes back as HTTPError, so each hop's headers can be read


def probe(url: str, sha256: str | None = None) -> Remote:
    """Size and sha256 of ``url``. The Hugging Face Hub puts both in the x-linked-* headers of its
    redirect (for LFS files); otherwise the size is the final Content-Length."""
    opener = urllib.request.build_opener(_NoRedirect)
    size: int | None = None
    current = url
    for _ in range(10):
        try:
            with opener.open(urllib.request.Request(current, method="HEAD", headers=HEADERS), timeout=TIMEOUT_S) as r:
                status, headers = r.status, r.headers
        except urllib.error.HTTPError as e:
            status, headers = e.code, e.headers
        if size is None and headers.get("x-linked-size"):
            size = int(headers["x-linked-size"])
        etag = (headers.get("x-linked-etag") or "").strip('"')
        if re.fullmatch(r"[0-9a-f]{64}", etag):
            sha256 = etag
        if status in (301, 302, 303, 307, 308) and headers.get("Location"):
            current = urljoin(current, headers["Location"])
            continue
        if status != 200:
            raise DownloadError(f"cannot get {url}: HTTP {status} (wrong file name?)")
        if size is None and headers.get("Content-Length"):
            size = int(headers["Content-Length"])
        if not size:
            raise DownloadError(f"cannot get {url}: the server does not say how big it is")
        return Remote(size, sha256)
    raise DownloadError(f"cannot get {url}: too many redirects")


def _size(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def _get(url: str, part: Path, progress: Progress, task) -> None:
    """Append the rest of ``url`` to ``part`` (an HTTP range request when it already has data)."""
    have = _size(part)
    headers = dict(HEADERS, **({"Range": f"bytes={have}-"} if have else {}))
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=TIMEOUT_S) as r:
        if have and r.status != 206:  # the server ignored the range: start this file over
            have = 0
            progress.update(task, completed=0)
        with open(part, "ab" if have else "wb") as f:
            while chunk := r.read(CHUNK):
                f.write(chunk)
                progress.advance(task, len(chunk))


def _sha256(path: Path, progress: Progress, name: str) -> str:
    h = hashlib.sha256()
    task = progress.add_task(f"  checking {name}", total=_size(path))
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
            progress.advance(task, len(chunk))
    progress.remove_task(task)
    return h.hexdigest()


def fetch_url(url: str, out: Path, progress: Progress, sha256: str | None = None) -> None:
    name = str(out.relative_to(MODELS)) if out.is_relative_to(MODELS) else str(out)
    if out.is_file():
        progress.console.print(f"  [dim]have[/dim]  {name}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    remote = probe(url, sha256)
    part = out.with_name(out.name + ".part")
    if _size(part) > remote.size:
        part.unlink()
    task = progress.add_task(f"  {name}", total=remote.size, completed=_size(part))
    stalls = 0
    while _size(part) < remote.size:
        before = _size(part)
        try:
            _get(url, part, progress, task)
            error = "connection closed early"
        except (OSError, http.client.HTTPException) as e:  # URLError/timeouts are OSErrors
            error = str(e) or type(e).__name__
        if _size(part) >= remote.size:
            break
        stalls = 0 if _size(part) > before else stalls + 1
        if stalls >= MAX_STALLS:
            raise DownloadError(f"{name}: no progress after {MAX_STALLS} attempts; run download again later to resume")
        progress.console.print(f"  [yellow]connection dropped[/yellow] ({error}), resuming in {RETRY_S} s…")
        time.sleep(RETRY_S)
    progress.remove_task(task)
    if _size(part) != remote.size:
        part.unlink()
        raise DownloadError(f"{name}: got more bytes than expected; deleted, run download again")
    if remote.sha256 and _sha256(part, progress, name) != remote.sha256:
        part.unlink()
        raise DownloadError(f"{name}: sha256 mismatch; deleted, run download again")
    part.rename(out)
    progress.console.print(f"  [green]got[/green]   {name} ({remote.size / 2**20:.0f} MiB)")


def fetch_hub(repo: str, file: str, folder: Path, progress: Progress) -> None:
    """A file from a Hugging Face model repo -> folder/file."""
    fetch_url(f"{HUB}/{repo}/resolve/main/{file}", folder / file, progress)


def run_download(console: Console, llm: str = DEFAULT_LLM, asr: str | None = None, only: str | None = None) -> bool:
    """Fetch every missing file; returns False if something could not be downloaded."""
    if llm not in LLMS:
        raise ValueError(f"unknown --llm {llm!r} ({' | '.join(LLMS)})")
    if asr not in (None, "large-v3"):
        raise ValueError(f"unknown --asr {asr!r} (large-v3)")
    if only not in (None, *GROUPS):
        raise ValueError(f"unknown --only {only!r} ({' | '.join(GROUPS)})")

    jobs: list[tuple[str, list[tuple]]] = []  # (heading, [fetch_hub/fetch_url arguments])
    if only in (None, "whisper"):
        turbo = MODELS / "whisper" / "faster-whisper-large-v3-turbo"
        jobs.append(("Speech recognition: Whisper large-v3-turbo (1.6 GB)",
                     [("mobiuslabsgmbh/faster-whisper-large-v3-turbo", f, turbo) for f in WHISPER_FILES]))
        if asr == "large-v3":
            big = MODELS / "whisper" / "faster-whisper-large-v3"
            jobs.append(("Speech recognition (optional): Whisper large-v3 (3.1 GB)",
                         [("Systran/faster-whisper-large-v3", f, big) for f in WHISPER_FILES]))
    if only in (None, "llm"):
        repo, file, _ = LLMS[llm]
        jobs.append((f"Translation: {repo} / {file}", [(repo, file, MODELS / "llm")]))
    if only in (None, "voices"):
        jobs.append(("Voices: Piper (3 Italian, 4 English)",
                     [("rhasspy/piper-voices", f, MODELS / "piper") for v in PIPER_VOICES for f in (f"{v}.json", v)]))
    if only in (None, "kokoro"):
        jobs.append(("Voices: Kokoro-82M (model + voice pack, 354 MB)",
                     [(f"{KOKORO_URL}/{f}", MODELS / "kokoro" / f, sha) for f, sha in KOKORO_FILES.items()]))

    failed: list[str] = []
    columns = (TextColumn("{task.description}"), BarColumn(), DownloadColumn(), TransferSpeedColumn(),
               TimeRemainingColumn())
    with Progress(*columns, console=console, transient=True) as progress:
        for heading, items in jobs:
            console.print(f"\n[bold]==> {heading}[/bold]")
            for item in items:
                try:
                    if item[0].startswith("https://"):
                        fetch_url(item[0], item[1], progress, item[2])
                    else:
                        fetch_hub(*item, progress)
                except DownloadError as e:
                    console.print(f"  [red]failed[/red] {e}")
                    failed.append(str(e))
    if llm != DEFAULT_LLM and only in (None, "llm"):
        _, file, style = LLMS[llm]
        console.print(f"\nNot the default model. To use it, set in config.toml [translation]:\n"
                      f'  model = "models/llm/{file}"\n  prompt_style = "{style}"')
    if asr == "large-v3":
        console.print("\nTo use Whisper large-v3: --set asr.model=models/whisper/faster-whisper-large-v3 "
                      "--set asr.compute_type=int8_float16 (keeps it inside 8 GB next to the LLM)")
    if failed:
        console.print(f"\n[red]{len(failed)} file(s) failed.[/red] Run [bold]./myvoice.py download[/bold] again "
                      "to resume; finished files are kept.")
        return False
    console.print("\n[green]All model files are here.[/green] Next: [bold]./myvoice.py check[/bold]")
    return True
