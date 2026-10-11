"""Command line interface (Typer + Rich). ``./myvoice.py`` with no command starts the conversation
app in auto mode (``-m``: manual mode)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from .config import Config, load_config

console = Console(highlight=False)
err_console = Console(stderr=True, highlight=False)

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
    help="Offline Italian ⇄ English speech translator.\n\n"
    "Without a command it starts the conversation app. [bold]Auto mode[/bold] (default, [bold]-a[/bold]) "
    "translates everything the microphone hears, Italian or English, and speaks it in the other language. "
    "[bold]Manual mode[/bold] ([bold]-m[/bold]): keys start and skip each step. [bold]TAB[/bold] switches "
    "between them in the app.",
)

# Options shared by several commands
ConfigOpt = Annotated[Optional[Path], typer.Option("--config", "-c", help="Config file (default: config.toml).")]
SetOpt = Annotated[
    Optional[list[str]],
    typer.Option("--set", metavar="SECTION.KEY=VALUE", help="Override a config value, e.g. --set asr.beam_size=1 (repeatable)."),
]
DirectionOpt = Annotated[Optional[str], typer.Option("--direction", "-d", help="auto | it-en | en-it (default: auto-detect).")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging (prompts, timings).")]
InputOpt = Annotated[Optional[str], typer.Option("--input-device", help="Microphone: index or name part (see `devices`).")]
OutputOpt = Annotated[Optional[str], typer.Option("--output-device", help="Speaker: index or name part (see `devices`).")]


class _DropChatty(logging.Filter):
    """Kokoro's phonemizer warns about harmless things ("words count mismatch", rare phonemes)
    and resets its own logger level on every call, so filter at the handler instead."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR or not record.name.startswith(("phonemizer", "kokoro_onnx"))


def _setup_logging(verbose: bool, quiet: bool = False) -> None:
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    handler = RichHandler(console=err_console, show_path=False, rich_tracebacks=False)
    if not verbose:
        handler.addFilter(_DropChatty())
    logging.basicConfig(level=level, format="%(message)s", datefmt="%H:%M:%S", handlers=[handler], force=True)
    for noisy in ("httpx", "urllib3", "huggingface_hub", "faster_whisper", "numba", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _config(config: Path | None, sets: list[str] | None, direction: str | None = None,
            input_device: str | None = None, output_device: str | None = None,
            extra: list[str] | None = None) -> Config:
    overrides = list(sets or [])
    if direction:
        overrides.append(f'general.direction="{direction}"')
    if input_device:
        overrides.append(f'audio.input_device="{input_device}"')
    if output_device:
        overrides.append(f'audio.output_device="{output_device}"')
    overrides += extra or []
    try:
        return load_config(config, overrides, local=True)
    except (ValueError, FileNotFoundError) as e:
        err_console.print(f"[red]config error:[/red] {e}")
        raise typer.Exit(2) from None


def _load_engine(cfg: Config, need_asr: bool = True, need_tts: bool = True, warmup: bool = True):
    from .pipeline import Engine

    with console.status("Loading models (speech recognition, translation, voices)…"):
        try:
            engine = Engine.load(cfg, need_asr=need_asr, need_tts=need_tts)
        except FileNotFoundError as e:
            err_console.print(f"[red]missing model:[/red] {e}\nRun [bold]./myvoice.py check[/bold] to see what is missing.")
            raise typer.Exit(1) from None
        if warmup and cfg.general.warmup:
            engine.warmup()
    return engine


TRANSLATE, SETUP = "Translate without the app", "Voices, setup and checks"
PASSTHROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


@app.callback(invoke_without_command=True)
def _start(
    ctx: typer.Context,
    auto: Annotated[bool, typer.Option("--auto", "-a", help="Auto mode (default): translate everything the microphone hears.")] = False,
    manual: Annotated[bool, typer.Option("--manual", "-m", help="Manual mode: keys start and skip each step (listen to them, your answer, ...).")] = False,
    headphones: Annotated[bool, typer.Option("--headphones", help="Auto mode keeps listening while a translation is spoken (no echo with headphones).")] = False,
    input_device: InputOpt = None,
    output_device: OutputOpt = None,
    config: ConfigOpt = None,
    set_: SetOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    if auto and manual:
        err_console.print("pick one: -a (auto mode) or -m (manual mode)")
        raise typer.Exit(2)
    _setup_logging(verbose, quiet=True)
    cfg = _config(config, set_, None, input_device, output_device, ["audio.half_duplex=false"] if headphones else None)
    from .talk import run_talk

    if not sys.stdin.isatty():
        err_console.print("the conversation app needs an interactive terminal; use [bold]file[/bold] or [bold]text[/bold] instead.")
        raise typer.Exit(2)
    engine = _load_engine(cfg)
    run_talk(engine, cfg, console, mode="manual" if manual else "auto")


@app.command("file", rich_help_panel=TRANSLATE)
def file_(
    input: Annotated[Path, typer.Argument(help="Recording to translate (wav, mp3, ogg/opus voice note, m4a, …).")],
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Also write the spoken translation to this WAV (+ .jsonl).")] = None,
    play: Annotated[bool, typer.Option("--play", "-p", help="Play the spoken translation.")] = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
    direction: DirectionOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Translate an audio file, e.g. an Italian voice message."""
    _setup_logging(verbose, quiet=True)
    cfg = _config(config, set_, direction)
    if not input.is_file():
        err_console.print(f"[red]no such file:[/red] {input}")
        raise typer.Exit(2)
    from .audio import Player, concat_with_gaps
    from .pipeline import translate_file

    engine = _load_engine(cfg, need_tts=output is not None or play, warmup=False)
    results = translate_file(engine, cfg, input, output, speak=play, printer=lambda m: console.print(m, markup=False))
    if play and results:
        audio, sr = concat_with_gaps([(r.audio, r.sample_rate) for r in results])
        Player(cfg.audio.output_device).play(audio, sr)
    if not results:
        console.print("[yellow]no speech found[/yellow]")


@app.command(rich_help_panel=TRANSLATE)
def text(
    words: Annotated[Optional[list[str]], typer.Argument(help="Text to translate (omit for an interactive prompt).")] = None,
    src: Annotated[Optional[str], typer.Option("--src", help="Source language (default: guessed).")] = None,
    speak: Annotated[bool, typer.Option("--speak", "-s", help="Also speak the translation.")] = False,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Write the spoken translation to this WAV.")] = None,
    config: ConfigOpt = None,
    set_: SetOpt = None,
    direction: DirectionOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Translate typed text (the language is detected; use --src to force it)."""
    _setup_logging(verbose, quiet=True)
    cfg = _config(config, set_, direction)
    from .audio import Player, write_wav
    from .pipeline import format_turn

    need_tts = speak or output is not None
    engine = _load_engine(cfg, need_asr=False, need_tts=need_tts, warmup=False)
    player = Player(cfg.audio.output_device) if speak else None

    def handle(line: str) -> None:
        r = engine.process_text(line, src=src, speak=need_tts)
        console.print(format_turn(r), markup=False)
        if r.audio.size:
            if output is not None:
                write_wav(output, r.audio, r.sample_rate)
                console.print(f"wrote {output}")
            if player is not None:
                player.play(r.audio, r.sample_rate)

    if words:
        handle(" ".join(words))
        return
    console.print("Type a sentence and press Enter (Ctrl+D to quit).")
    for line in sys.stdin:
        if line.strip():
            handle(line.strip())


@app.command(rich_help_panel=SETUP)
def voices(
    use: Annotated[Optional[list[str]], typer.Option("--use", help="Make this voice the default for its language (saved), e.g. --use kokoro:im_nicola.")] = None,
    play: Annotated[bool, typer.Option("--play", "-p", help="Speak a sample with every listed voice.")] = False,
    all_: Annotated[bool, typer.Option("--all", help="Also list every Kokoro voice, not only the recommended ones.")] = False,
    config: ConfigOpt = None,
    set_: SetOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """List the installed voices (and pick the default ones)."""
    _setup_logging(verbose, quiet=True)
    cfg = _config(config, set_)
    from .config import save_local
    from .tts import list_voices, voice_language

    for voice_id in use or []:
        lang = voice_language(cfg, voice_id)
        if lang not in cfg.general.languages:
            err_console.print(f"[red]unknown voice[/red] {voice_id!r}; pick one from the list below")
            raise typer.Exit(2)
        save_local(cfg, {"tts": {"voices": {lang: voice_id}}})
        console.print(f"{lang}: now using [bold]{voice_id}[/bold] (saved to config.local.toml)")

    table = Table(title="Installed voices", title_justify="left")
    for col in ("", "id", "language", "voice", "engine"):
        table.add_column(col)
    order = [cfg.talk.their_language, cfg.talk.my_language]
    found = sorted((v for v in list_voices(cfg, all_kokoro=all_) if v.lang in order),
                   key=lambda v: (order.index(v.lang), v.engine != "kokoro", v.id))
    for v in found:
        current = cfg.tts.voices.get(v.lang) == v.id
        table.add_row("★" if current else "", v.id, v.lang, f"{v.name} ({v.detail})", v.engine,
                      style="bold" if current else None)
    console.print(table)
    console.print("[dim]★ = in use. Change with [bold]--use ID[/bold], or press V in the app.[/dim]")
    if play:
        from .audio import Player
        from .talk import SAMPLE_SENTENCES
        from .tts import Tts

        tts = Tts(cfg)
        player = Player(cfg.audio.output_device)
        for v in found:
            console.print(f"  ▶ {v.id}")
            audio, sr = tts.synthesize(SAMPLE_SENTENCES.get(v.lang, "Ok."), v.lang, voice_id=v.id)
            player.play(audio, sr)


@app.command(rich_help_panel=SETUP)
def check(config: ConfigOpt = None, set_: SetOpt = None) -> None:
    """Verify the environment, CUDA libraries and model files (no models are loaded)."""
    _setup_logging(False, quiet=True)
    cfg = _config(config, set_)
    from .check import run_check

    if not run_check(cfg):
        raise typer.Exit(1)


@app.command(rich_help_panel=SETUP)
def devices() -> None:
    """List audio input/output devices."""
    from .audio import list_devices

    console.print(list_devices())


@app.command(rich_help_panel=SETUP)
def selftest(
    out_dir: Annotated[Optional[Path], typer.Option("--out-dir", help="Where to write the WAVs (default: outputs/).")] = None,
    config: ConfigOpt = None,
    set_: SetOpt = None,
    direction: DirectionOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """End-to-end test without a microphone: a voice speaks a sentence, then it is translated."""
    _setup_logging(verbose)
    cfg = _config(config, set_, direction)
    from .pipeline import selftest as run_selftest

    engine = _load_engine(cfg)
    if not run_selftest(engine, cfg, out_dir or cfg.path(cfg.general.session_log_dir or "outputs")):
        raise typer.Exit(1)


@app.command(rich_help_panel=SETUP, context_settings={**PASSTHROUGH, "help_option_names": []})
def setup(ctx: typer.Context) -> None:
    """Create the Python environment and install the packages (first time, or to repair it). See [bold]setup --help[/bold]."""
    from .install import main as setup_main

    raise typer.Exit(setup_main(ctx.args))


@app.command(rich_help_panel=SETUP)
def download(
    llm: Annotated[str, typer.Option("--llm", help="translategemma-q6 (default) | translategemma-q4 | translategemma-q8 | qwen3.5-4b | gemma4-e4b")] = "translategemma-q6",
    asr: Annotated[Optional[str], typer.Option("--asr", help="Also get Whisper large-v3 (3.1 GB, optional; see to-download.md).")] = None,
    only: Annotated[Optional[str], typer.Option("--only", help="Just one group: whisper | llm | voices | kokoro.")] = None,
) -> None:
    """Download the models into ./models (skips files that are already there; resumes broken downloads)."""
    from .download import run_download

    try:
        ok = run_download(console, llm=llm, asr=asr, only=only)
    except ValueError as e:
        err_console.print(f"[red]{e}[/red]")
        raise typer.Exit(2) from None
    if not ok:
        raise typer.Exit(1)


@app.command(rich_help_panel=SETUP, context_settings=PASSTHROUGH)
def test(ctx: typer.Context) -> None:
    """Run the unit tests (fake models, no GPU needed); extra arguments go to pytest."""
    import pytest

    from .config import PROJECT_ROOT

    raise typer.Exit(pytest.main(["-q", str(PROJECT_ROOT / "tests"), *ctx.args]))


def main(argv: list[str] | None = None) -> None:
    try:
        app(args=argv, prog_name="./myvoice.py")
        code: object = 0
    except SystemExit as e:
        code = e.code
    # Leave without running the native libraries' static destructors: with CTranslate2, llama.cpp,
    # onnxruntime and PortAudio threads all loaded, their teardown order at exit is racy and can
    # abort with "terminate called without an active exception". Everything of ours (session logs,
    # terminal mode) is closed by now; flush and exit directly.
    if isinstance(code, str):
        print(code, file=sys.stderr)
        code = 1
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code if isinstance(code, int) else 0)
