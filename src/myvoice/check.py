"""`check` command: verify the environment, GPU libraries and model files without loading models."""

from __future__ import annotations

import importlib
import os
import platform
import signal
import subprocess
import sys
from pathlib import Path
from typing import Callable

from .config import Config

OK, WARN, FAIL = "ok", "warn", "FAIL"


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, what: str, detail: str = "") -> None:
        self.rows.append((status, what, detail))

    def guard(self, what: str, fn: Callable[[], tuple[str, str]], fail_status: str = FAIL) -> None:
        try:
            status, detail = fn()
        except Exception as e:  # noqa: BLE001 - every failure becomes a report row
            status, detail = fail_status, f"{type(e).__name__}: {e}"
        self.add(status, what, detail)

    def print(self) -> bool:
        width = max(len(r[1]) for r in self.rows)
        for status, what, detail in self.rows:
            tag = {OK: " ok ", WARN: "WARN", FAIL: "FAIL"}[status]
            print(f"[{tag}] {what:<{width}}  {detail}")
        fails = sum(r[0] == FAIL for r in self.rows)
        warns = sum(r[0] == WARN for r in self.rows)
        print(f"\n{len(self.rows)} checks: {fails} failed, {warns} warnings")
        return fails == 0


def _dist_version(dist: str) -> str:
    import importlib.metadata

    return importlib.metadata.version(dist)


def _version(mod: str) -> tuple[str, str]:
    m = importlib.import_module(mod)
    return OK, str(getattr(m, "__version__", "installed"))


def _torch() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        return FAIL, f"torch {torch.__version__} but CUDA is NOT available"
    free, total = torch.cuda.mem_get_info()
    return OK, (
        f"torch {torch.__version__} (CUDA {torch.version.cuda}), {torch.cuda.get_device_name(0)}, "
        f"{free / 2**30:.1f}/{total / 2**30:.1f} GiB free"
    )


def _ctranslate2() -> tuple[str, str]:
    import ctranslate2

    n = ctranslate2.get_cuda_device_count()
    if n == 0:
        return FAIL, f"ctranslate2 {ctranslate2.__version__} sees no CUDA device (cuBLAS/cuDNN not found? source env.sh)"
    types = sorted(ctranslate2.get_supported_compute_types("cuda"))
    return OK, f"ctranslate2 {ctranslate2.__version__}, compute types: {', '.join(types)}"


def _cudnn_for_ct2() -> tuple[str, str]:
    import ctypes

    for name in ("libcudnn.so.9", "libcublas.so.12"):
        ctypes.CDLL(name)
    return OK, "libcudnn.so.9 and libcublas.so.12 loadable"


def _llama_cpp() -> tuple[str, str]:
    # In a child process: a llama.cpp build for another CPU (e.g. AVX-512) dies with SIGILL
    # instead of raising, which would take the whole report down with it.
    code = "import llama_cpp; print(llama_cpp.__version__, llama_cpp.llama_supports_gpu_offload())"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    if p.returncode == -signal.SIGILL:
        return FAIL, "llama-cpp-python crashed (illegal instruction): built for another CPU; run `bash scripts/setup_env.sh`"
    if p.returncode != 0:
        last = (p.stderr.strip().splitlines() or ["no output"])[-1]
        return FAIL, f"llama-cpp-python failed to load (exit {p.returncode}): {last}"
    version, gpu = p.stdout.split()[-2:]
    if gpu != "True":
        return FAIL, f"llama-cpp-python {version} was built WITHOUT GPU support (see to-download.md)"
    return OK, f"llama-cpp-python {version} with GPU offload"


def _piper() -> tuple[str, str]:
    import onnxruntime
    from piper import PiperVoice  # noqa: F401

    return OK, f"piper ok, onnxruntime {onnxruntime.__version__} providers: {', '.join(onnxruntime.get_available_providers())}"


def _kokoro() -> tuple[str, str]:
    import importlib.metadata

    importlib.import_module("kokoro_onnx")
    return OK, f"kokoro-onnx {importlib.metadata.version('kokoro-onnx')}"


def _voice(cfg: Config, voice_id: str) -> Callable[[], tuple[str, str]]:
    def check() -> tuple[str, str]:
        from .tts import KOKORO_PREFIX, resolve_piper

        if voice_id.startswith(KOKORO_PREFIX):
            import numpy as np

            model, pack = cfg.path(cfg.tts.kokoro_model), cfg.path(cfg.tts.kokoro_voices)
            for f in (model, pack):
                if not f.is_file():
                    return FAIL, f"{voice_id}: missing {f} (run scripts/download_models.sh)"
            with np.load(pack) as voices:
                if voice_id[len(KOKORO_PREFIX):] not in voices.files:
                    return FAIL, f"{voice_id}: not in {pack.name}"
            return OK, f"{voice_id} (Kokoro, {model.stat().st_size / 2**20:.0f} MiB model)"
        try:
            onnx = resolve_piper(cfg, voice_id)
        except FileNotFoundError as e:
            return FAIL, str(e)
        if not onnx.is_file() or not Path(f"{onnx}.json").is_file():
            return FAIL, f"{voice_id}: missing {onnx} or its .json (run scripts/download_models.sh)"
        return OK, f"{voice_id} (Piper, {onnx.stat().st_size / 2**20:.0f} MiB)"

    return check


def _silero() -> tuple[str, str]:
    from silero_vad import load_silero_vad

    load_silero_vad()
    return OK, "silero VAD model (bundled with the package) loads"


def _sounddevice() -> tuple[str, str]:
    import sounddevice as sd

    default_in, default_out = sd.default.device
    devs = sd.query_devices()
    n_in = sum(d["max_input_channels"] > 0 for d in devs)
    n_out = sum(d["max_output_channels"] > 0 for d in devs)
    if n_in == 0:
        return WARN, f"PortAudio {sd.get_portaudio_version()[1]}: no input devices"
    return OK, f"{n_in} input / {n_out} output devices (default in={default_in}, out={default_out})"


def _file(path: Path, min_bytes: int = 1) -> Callable[[], tuple[str, str]]:
    def check() -> tuple[str, str]:
        if not path.exists():
            return FAIL, f"missing: {path}"
        size = path.stat().st_size
        if size < min_bytes:
            return FAIL, f"{path} is only {size} bytes (incomplete download?)"
        return OK, f"{path} ({size / 2**20:.0f} MiB)"

    return check


def run_check(cfg: Config) -> bool:
    r = Report()
    r.add(OK, "python", f"{sys.version.split()[0]} on {platform.system()} ({sys.executable})")
    if os.environ.get("MYVOICE_ENV_SOURCED") != "1":
        r.add(WARN, "env.sh", "not sourced; use ./run.sh or `source env.sh` (it sources setvars.all.sh)")
    else:
        r.add(OK, "env.sh", "sourced")
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    r.add(OK if offline else WARN, "offline mode", "HF_HUB_OFFLINE=1" if offline else "HF_HUB_OFFLINE is not 1")

    r.guard("numpy", lambda: _version("numpy"))
    r.guard("torch + CUDA", _torch)
    r.guard("faster-whisper", lambda: _version("faster_whisper"))
    r.guard("ctranslate2 CUDA", _ctranslate2)
    r.guard("cuDNN/cuBLAS libs", _cudnn_for_ct2)
    if cfg.translation.backend == "llama_cpp":
        r.guard("llama-cpp-python", _llama_cpp)
    else:
        r.guard("transformers", lambda: _version("transformers"))
        if cfg.translation.load_in_4bit:
            r.guard("bitsandbytes", lambda: _version("bitsandbytes"))
    r.guard("piper-tts", _piper)
    uses_kokoro = any(v.startswith("kokoro:") for v in cfg.tts.voices.values())
    r.guard("kokoro-onnx", _kokoro, fail_status=FAIL if uses_kokoro else WARN)
    r.guard("rich + typer (UI)", lambda: (OK, f"rich {_dist_version('rich')}, typer {_dist_version('typer')}"))
    if cfg.vad.backend == "silero":
        r.guard("silero-vad", _silero)
    r.guard("jinja2", lambda: _version("jinja2"))
    r.guard("sounddevice/PortAudio", _sounddevice, fail_status=WARN)

    r.guard("model: whisper", _file(cfg.path(cfg.asr.model) / "model.bin", 100 * 2**20))
    if cfg.translation.backend == "llama_cpp":
        r.guard("model: LLM (GGUF)", _file(cfg.path(cfg.translation.model), 500 * 2**20))
    else:
        ref = cfg.model_ref(cfg.translation.model)
        if Path(ref).exists():
            r.guard("model: LLM (HF dir)", _file(Path(ref) / "config.json"))
        else:
            r.add(OK, "model: LLM (HF cache)", f"{ref} (resolved from the Hugging Face cache at load time)")
    for lang in cfg.general.languages:
        r.guard(f"voice: {lang}", _voice(cfg, cfg.tts.voices[lang]))
    return r.print()
