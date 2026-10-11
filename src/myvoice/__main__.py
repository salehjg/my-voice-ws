"""Command line interface: ``python -m myvoice <command>`` (normally via ``./run.sh``)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import load_config


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", help="config file (default: ./config.toml)")
    common.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
        help="override a config value, e.g. --set asr.compute_type=int8_float16 (repeatable)",
    )
    common.add_argument("--mode", help="auto | it-en | en-it (overrides general.mode)")
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging (prompts, timings)")

    p = argparse.ArgumentParser(
        prog="myvoice",
        description="Offline Italian <-> English speech translator (Whisper -> LLM -> Piper).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", parents=[common], help="verify environment, CUDA libraries and model files")
    sub.add_parser("devices", parents=[common], help="list audio input/output devices")

    live = sub.add_parser("live", parents=[common], help="real-time translation: microphone -> speaker")
    live.add_argument("--input-device", help="mic index or name substring")
    live.add_argument("--output-device", help="speaker index or name substring")
    live.add_argument("--headphones", action="store_true", help="full duplex: don't mute the mic during playback")

    f = sub.add_parser("file", parents=[common], help="translate a recording (any format ffmpeg/PyAV reads)")
    f.add_argument("input", type=Path)
    f.add_argument("-o", "--output", type=Path, help="write translated speech to this WAV (+ .jsonl transcript)")

    t = sub.add_parser("text", parents=[common], help="translate typed text (interactive if no text given)")
    t.add_argument("text", nargs="*")
    t.add_argument("--src", help="source language (default: guessed from the text)")
    t.add_argument("--speak", action="store_true", help="also speak the translation")
    t.add_argument("-o", "--output", type=Path, help="write the spoken translation to this WAV")

    s = sub.add_parser("selftest", parents=[common], help="end-to-end test without a microphone (Piper -> pipeline)")
    s.add_argument("--out-dir", type=Path, default=None, help="where to write WAVs (default: outputs/)")
    return p


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "urllib3", "huggingface_hub", "faster_whisper", "numba", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    overrides = list(args.overrides)
    if args.mode:
        overrides.append(f'general.mode="{args.mode}"')
    if getattr(args, "input_device", None):
        overrides.append(f'audio.input_device="{args.input_device}"')
    if getattr(args, "output_device", None):
        overrides.append(f'audio.output_device="{args.output_device}"')
    if getattr(args, "headphones", False):
        overrides.append("audio.half_duplex=false")
    try:
        cfg = load_config(args.config, overrides)
    except (ValueError, FileNotFoundError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    if args.command == "check":
        from .check import run_check

        return 0 if run_check(cfg) else 1

    if args.command == "devices":
        from .audio import list_devices

        print(list_devices())
        return 0

    from .pipeline import Engine

    if args.command == "live":
        from .pipeline import LiveSession

        engine = Engine.load(cfg)
        if cfg.general.warmup:
            engine.warmup()
        LiveSession(engine, cfg).run()
        return 0

    if args.command == "file":
        from .pipeline import translate_file

        if not args.input.exists():
            print(f"no such file: {args.input}", file=sys.stderr)
            return 2
        engine = Engine.load(cfg, need_tts=args.output is not None)
        translate_file(engine, cfg, args.input, args.output)
        return 0

    if args.command == "text":
        from .audio import Player, write_wav
        from .pipeline import format_turn

        speak = args.speak or args.output is not None
        engine = Engine.load(cfg, need_asr=False, need_tts=speak)
        player = Player(cfg.audio.output_device) if args.speak else None

        def handle(line: str) -> None:
            r = engine.process_text(line, src=args.src, speak=speak)
            print(format_turn(r))
            if r.audio.size:
                if args.output is not None:
                    write_wav(args.output, r.audio, r.sample_rate)
                    print(f"wrote {args.output}")
                if player is not None:
                    player.play(r.audio, r.sample_rate)

        if args.text:
            handle(" ".join(args.text))
        else:
            print("Type a sentence and press Enter (Ctrl+D to quit).")
            for line in sys.stdin:
                if line.strip():
                    handle(line.strip())
        return 0

    if args.command == "selftest":
        from .pipeline import selftest

        out_dir = args.out_dir or cfg.path(cfg.general.session_log_dir or "outputs")
        engine = Engine.load(cfg)
        if cfg.general.warmup:
            engine.warmup()
        return 0 if selftest(engine, cfg, out_dir) else 1

    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
