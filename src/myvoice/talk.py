"""Conversation mode: they speak (Italian), you read/hear it in English, you answer in English
and it is spoken in Italian. Single keys drive the cycle; every step can be skipped.

    ready ──SPACE──► listen to them ─► translate ─► (speak to you) ─► your turn
      ▲                                                                   │ SPACE
      └─────────── speak your answer in their language ◄─ confirm ◄─ record you
"""

from __future__ import annotations

import dataclasses
import logging
import os
import queue
import re
import select
import sys
import time
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .audio import Player, concat_with_gaps, open_mic, read_audio
from .config import Config, save_local
from .langs import get_language
from .pipeline import Engine, SessionLog, TurnResult
from .vad import SAMPLE_RATE, Segmenter, make_vad

log = logging.getLogger(__name__)

THEIR_COLOR, MY_COLOR = "cyan", "green"
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=\S)")
AGAIN = "\x00again"  # confirm(): record again
KNOWN_KEYS = {"space", "enter", "e", "l", "t", "i", "f", "s", "esc", "r", "a", "c", "v", "h", "?", "q", "ctrl-d"}

SAMPLE_SENTENCES = {
    "it": "Ciao! Questa è la voce italiana. Come ti sembra?",
    "en": "Hi! This is the English voice. How does it sound?",
}


# --------------------------------------------------------------------------- keys


class Keys(Protocol):
    def get(self, timeout: float | None = None) -> str | None: ...
    def flush(self) -> None: ...
    def line_input(self, prompt: str, initial: str = "") -> str | None: ...


def decode_key(data: bytes) -> str | None:
    """Raw terminal bytes -> key name: "space", "enter", "esc", "backspace", "up", or a character."""
    if not data:
        return None
    if data in (b"\r", b"\n", b"\r\n"):
        return "enter"
    if data == b" ":
        return "space"
    if data == b"\x1b":
        return "esc"
    if data.startswith(b"\x1b["):
        return {b"A": "up", b"B": "down", b"C": "right", b"D": "left"}.get(data[2:3])
    if data in (b"\x7f", b"\x08"):
        return "backspace"
    if data == b"\x04":
        return "ctrl-d"
    try:
        ch = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    return ch[0].lower() if ch[:1].isprintable() else None


class TerminalKeys:
    """Single keypresses from the terminal (cbreak mode, restored on exit)."""

    def __init__(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("the talk app needs an interactive terminal (try `file` or `text` instead)")
        self.fd = sys.stdin.fileno()
        self._saved = None

    def __enter__(self) -> TerminalKeys:
        import termios
        import tty

        self._saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc) -> None:
        import termios

        if self._saved is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)

    def get(self, timeout: float | None = None) -> str | None:
        ready, _, _ = select.select([self.fd], [], [], timeout)
        if not ready:
            return None
        data = os.read(self.fd, 32)
        if data.startswith(b"\x1b") and len(data) == 1:
            # a lone ESC may be the start of an arrow-key sequence that hasn't arrived yet
            more, _, _ = select.select([self.fd], [], [], 0.03)
            if more:
                data += os.read(self.fd, 32)
        return decode_key(data)

    def flush(self) -> None:
        import termios

        termios.tcflush(self.fd, termios.TCIFLUSH)

    def line_input(self, prompt: str, initial: str = "") -> str | None:
        """Read a line in normal (cooked) mode, pre-filled with ``initial``. None on Ctrl+C/Ctrl+D."""
        import readline
        import termios

        termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
        readline.set_startup_hook(lambda: readline.insert_text(initial))
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        finally:
            readline.set_startup_hook(None)
            import tty

            tty.setcbreak(self.fd)


class ScriptedKeys:
    """Keys from a list (tests, demos). Short polls (timeout < 0.5 s) never consume a key, so
    recordings end by silence and playback runs to the end; ``line_input`` takes the next item."""

    def __init__(self, keys: list[str], idle: Callable[[], None] | None = None) -> None:
        self.keys = list(keys)
        self.idle = idle

    def get(self, timeout: float | None = None) -> str | None:
        if timeout is not None and timeout < 0.5:
            if self.idle:
                self.idle()
            return None
        return self.keys.pop(0) if self.keys else "q"

    def flush(self) -> None:
        pass

    def line_input(self, prompt: str, initial: str = "") -> str | None:
        if not self.keys:
            return None
        item = self.keys.pop(0)
        return None if item == "<cancel>" else item.replace("<initial>", initial)


# --------------------------------------------------------------------------- recording


@dataclasses.dataclass
class Recording:
    audio: np.ndarray
    stopped_by: str  # "silence" | "key" | "limit"

    @property
    def seconds(self) -> float:
        return self.audio.size / SAMPLE_RATE


def level_db(block: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(block)))) if block.size else 0.0
    return 20.0 * np.log10(rms + 1e-9)


def record_utterance(
    cfg: Config,
    keys: Keys,
    end_silence_ms: int,
    mic,
    on_update: Callable[[float, float, bool], None] = lambda db, t, speaking: None,
) -> Recording | None:
    """Record one utterance from ``mic``: from the first speech until ``end_silence_ms`` of silence.

    SPACE/ENTER stops early and keeps what was said, even if the voice detector never triggered
    (quiet mic): then everything recorded so far is used. ESC cancels. Returns None when
    cancelled or when nobody spoke within ``talk.no_speech_timeout_s``.
    """
    vad_cfg = dataclasses.replace(cfg.vad, min_silence_ms=end_silence_ms, max_utterance_s=cfg.talk.max_record_s)
    segmenter = Segmenter(make_vad(vad_cfg), vad_cfg)
    keys.flush()
    while not mic.blocks.empty():  # leftovers from the previous recording
        mic.blocks.get_nowait()
    mic.start()
    t0 = time.monotonic()
    heard_speech = False
    raw: list[np.ndarray] = []  # everything, for a manual stop the VAD didn't see coming
    try:
        while True:
            key = keys.get(timeout=0)
            if key == "esc":
                return None
            if key in ("space", "enter"):
                utt = segmenter.flush()
                if utt is not None:
                    return Recording(utt.audio, "key")
                audio = np.concatenate(raw) if raw else np.zeros(0, np.float32)
                return Recording(audio, "key") if audio.size > SAMPLE_RATE // 2 else None
            try:
                block = mic.blocks.get(timeout=0.05)
            except queue.Empty:
                block = None
            elapsed = time.monotonic() - t0
            if block is not None:
                if elapsed <= cfg.talk.max_record_s:
                    raw.append(block)
                for utt in segmenter.feed(block):
                    return Recording(utt.audio, "limit" if utt.forced_cut else "silence")
                heard_speech = heard_speech or segmenter.in_speech
                on_update(level_db(block), elapsed, segmenter.in_speech)
            if not heard_speech and elapsed > cfg.talk.no_speech_timeout_s:
                return None
    finally:
        mic.stop()


# --------------------------------------------------------------------------- app


def _key(name: str) -> Text:
    return Text(f" {name} ", style="bold reverse")


def _hint(*items: tuple[str, str]) -> Text:
    t = Text()
    for i, (key, what) in enumerate(items):
        if i:
            t.append("   ")
        t.append_text(_key(key))
        t.append(f" {what}")
    return t


def _timings(r: TurnResult) -> str:
    return " · ".join(f"{k} {v:.1f}s" for k, v in r.timings.items())


class TalkApp:
    def __init__(
        self,
        engine: Engine,
        cfg: Config,
        keys: Keys,
        console: Console | None = None,
        player: Player | None = None,
        mic_factory: Callable[[], object] | None = None,
        save_choices: bool = True,
    ) -> None:
        self.engine = engine
        self.cfg = cfg
        self.keys = keys
        self.console = console or Console(highlight=False)
        self.player = player or Player(cfg.audio.output_device)
        self.mic = (mic_factory or (lambda: open_mic(cfg.audio.sample_rate, cfg.audio.block_ms,
                                                       cfg.audio.input_device)))()
        self.save_choices = save_choices
        self.their = cfg.talk.their_language
        self.mine = cfg.talk.my_language
        self.their_name = get_language(self.their).name
        self.my_name = get_language(self.mine).name
        self.log = SessionLog(cfg, prefix="talk")
        self.last_audio: tuple[np.ndarray, int] | None = None
        self.history: list[TurnResult] = []

    # -- output helpers
    def say(self, text: str | Text, style: str = "") -> None:
        self.console.print(text, style=style)

    def hint(self, *items: tuple[str, str]) -> None:
        self.console.print(_hint(*items), style="dim")

    def status_line(self) -> Text:
        on = lambda b: ("on", "bold green") if b else ("off", "bold red")  # noqa: E731
        tts = self.engine.tts
        voices = ""
        if tts is not None:
            voices = f"{self._voice_name(tts.voices[self.their])} / {self._voice_name(tts.voices[self.mine])}"
        return Text.assemble(
            (f"{self.my_name} audio ", "dim"), on(self.cfg.talk.speak_to_me),
            ("   confirm my speech ", "dim"), on(self.cfg.talk.confirm_my_speech),
            ("   voices ", "dim"), (voices, "bold"),
        )

    def _voice_name(self, voice_id: str) -> str:
        for v in self._voices_cache():
            if v.id == voice_id:
                return v.name
        return voice_id.split(":")[-1]

    def _voices_cache(self):
        from .tts import list_voices

        if not hasattr(self, "_voices"):
            self._voices = list_voices(self.cfg)
        return self._voices

    def banner(self) -> None:
        title = f"myvoice · {self.their_name} ⇄ {self.my_name}"
        body = Group(
            Text(f"They speak {self.their_name}, you read/hear {self.my_name}; you answer in {self.my_name}, "
                 f"they hear {self.their_name}."),
            Text(),
            _hint(("SPACE", "next step"), ("H", "all keys"), ("Q", "quit")),
            self.status_line(),
        )
        self.console.print(Panel(body, title=title, title_align="left", border_style="blue"))

    def help(self) -> None:
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column(style="bold reverse", justify="center")
        t.add_column()
        rows = [
            ("SPACE", f"next step: listen to them → your answer → speak it in {self.their_name}"),
            ("E", f"answer in {self.my_name} by voice (skip listening to them)"),
            ("T", f"type in {self.my_name} (translated + spoken in {self.their_name})"),
            ("I", f"type in {self.their_name} (translated into {self.my_name})"),
            ("L", "listen to them (again)"),
            ("F", "translate an audio file (voice message, recording)"),
            ("R", "replay the last translation"),
            ("A", f"{self.my_name} audio on/off (hear the translation, not only read it)"),
            ("C", "confirm my speech on/off (check what was heard before translating)"),
            ("V", "change voices"),
            ("S / ESC", "skip / cancel the current step"),
            ("Q", "quit"),
        ]
        for k, what in rows:
            t.add_row(k, what)
        t.add_row("", "")
        t.add_row("", Text("While recording: SPACE/ENTER = done, ESC = cancel. While speaking: any key = stop.",
                           style="dim"))
        self.console.print(Panel(t, title="keys", title_align="left", border_style="blue"))

    # -- audio
    @staticmethod
    def _playing_view(label: str) -> Text:
        return Text.assemble(("  🔊 ", ""), (label, "bold"), ("   any key = stop", "dim"))

    def _wait_playing(self) -> bool:
        """Wait for the current sound to end; False if a key stopped it."""
        while self.player.is_playing():
            if self.keys.get(timeout=0.05) is not None:
                self.player.stop()
                return False
        return True

    def play(self, audio: np.ndarray, sr: int, label: str) -> None:
        """Play in the background; any key stops it."""
        if audio.size == 0:
            return
        self.last_audio = (audio, sr)
        self.keys.flush()
        self.player.start(audio, sr)
        with Live(self._playing_view(label), console=self.console, transient=True, refresh_per_second=8):
            self._wait_playing()
        self.keys.flush()

    def record(self, lang: str, end_silence_ms: int) -> Recording | None:
        name = get_language(lang).name
        color = THEIR_COLOR if lang == self.their else MY_COLOR
        view = Text()

        def update(db: float, elapsed: float, speaking: bool) -> None:
            n = int(max(0.0, min(1.0, (db + 60) / 50)) * 24)
            view.truncate(0)
            view.append(" ● ", style="bold red" if speaking else "red")
            view.append(f"Listening to {name}… ", style=f"bold {color}")
            view.append("█" * n, style="green" if speaking else "grey50")
            view.append("░" * (24 - n), style="grey23")
            view.append(f" {elapsed:4.1f}s   ")
            view.append_text(_hint(("SPACE", "done"), ("ESC", "cancel")))
            live.update(view)

        with Live(Text(f" ● Listening to {name}…", style=f"bold {color}"), console=self.console,
                  transient=True, refresh_per_second=15) as live:
            rec = record_utterance(self.cfg, self.keys, end_silence_ms, self.mic, update)
        if rec is None:
            self.say("  (nothing recorded — if you did speak, watch the level bar; SPACE stops and keeps it)", "dim")
        elif rec.stopped_by == "limit":
            self.say(f"  (stopped at the {self.cfg.talk.max_record_s:.0f}s limit)", "dim")
        return rec

    # -- turns
    def show(self, r: TurnResult, who: str) -> None:
        theirs = r.src_lang == self.their
        color = THEIR_COLOR if theirs else MY_COLOR
        body = Group(
            Text(r.src_text, style="dim"),
            Text.assemble(("→ ", "bold"), (r.tgt_text, f"bold {color}")),
        )
        title = f"{who} · {get_language(r.src_lang).name} → {get_language(r.tgt_lang).name}"
        self.console.print(Panel(body, title=title, title_align="left", subtitle=_timings(r),
                                 subtitle_align="right", border_style=color))
        self.history.append(r)
        self.log.write(r)

    def their_turn(self) -> str:
        rec = self.record(self.their, self.cfg.talk.their_end_silence_ms)
        if rec is None:
            return "ready"
        with self.console.status("Translating…"):
            r = self.engine.process_audio(rec.audio, speak=False, src=self.their)
        if r is None or not r.tgt_text:
            self.say(f"  Didn't catch any {self.their_name} — press SPACE to listen again.", "yellow")
            return "ready"
        self.show(r, "They said")
        if self.cfg.talk.speak_to_me:
            self.speak(r.tgt_text, r.tgt_lang, f"{self.my_name} translation")
        return "reply"

    def speak(self, text: str, lang: str, label: str) -> None:
        """Speak ``text`` (already on screen) sentence by sentence: the next sentence is synthesized
        while the previous one plays, so the voice starts sooner. Any key stops it."""
        tts = self.engine.tts
        if tts is None or not text.strip():
            return
        parts = [p for p in SENTENCE_END.split(text.strip()) if p.strip()]
        gap_s = self.cfg.tts.sentence_silence_ms / 1000
        with self.console.status("…", spinner="simpleDots"):
            clips = [tts.synthesize(parts[0], lang)]
        self.keys.flush()
        with Live(self._playing_view(label), console=self.console, transient=True, refresh_per_second=8):
            self.player.start(*clips[0])
            for part in parts[1:]:
                audio, sr = tts.synthesize(part, lang)  # while the previous sentence plays
                audio = np.concatenate([np.zeros(int(gap_s * sr), np.float32), audio])
                clips.append((audio, sr))
                if not self._wait_playing():
                    break
                self.player.start(audio, sr)
            else:
                self._wait_playing()
        self.last_audio = concat_with_gaps(clips, gap_s=0.0)
        self.keys.flush()

    def my_turn(self) -> str:
        while True:
            rec = self.record(self.mine, self.cfg.talk.my_end_silence_ms)
            if rec is None:
                return "reply"
            with self.console.status("Transcribing…"):
                t0 = time.perf_counter()
                tr = self.engine.transcribe(rec.audio, self.mine)
                asr_s = time.perf_counter() - t0
            if tr is None:
                self.say("  Didn't catch that — press SPACE to try again, T to type.", "yellow")
                return "reply"
            text: str | None = tr.text
            if self.cfg.talk.confirm_my_speech:
                text = self.confirm(tr.text)
                if text == AGAIN:
                    continue
                if text is None:
                    return "reply"
            return self.send_mine(text, {"asr": asr_s})

    def confirm(self, text: str) -> str | None:
        """Show what was heard; ENTER accepts, E edits, R re-records (returns AGAIN), ESC cancels."""
        while True:
            self.console.print(Text.assemble(("  You said: ", "dim"), (text, f"bold {MY_COLOR}")))
            self.hint(("ENTER", f"translate + speak in {self.their_name}"), ("E", "edit"), ("R", "record again"),
                      ("ESC", "cancel"))
            key = self.keys.get()
            if key in ("enter", "space"):
                return text
            if key == "e":
                edited = self.keys.line_input("  edit> ", text)
                if edited is not None and edited.strip():
                    text = edited.strip()
                    return text
                continue
            if key == "r":
                return AGAIN
            if key in ("esc", "s", "q"):
                self.say("  (cancelled)", "dim")
                return None

    def send_mine(self, text: str, timings: dict[str, float] | None = None) -> str:
        with self.console.status(f"Translating into {self.their_name}…"):
            r = self.engine.process_text(text, src=self.mine, speak=False)
        r.timings = {**(timings or {}), **r.timings}
        self.show(r, "You said")
        self.speak(r.tgt_text, r.tgt_lang, f"speaking {self.their_name}")
        return "ready"

    def type_text(self, lang: str) -> str:
        name = get_language(lang).name
        text = self.keys.line_input(f"  {name}> ")
        if not text or not text.strip():
            return "ready" if lang == self.their else "reply"
        if lang == self.mine:
            return self.send_mine(text.strip())
        with self.console.status("Translating…"):
            r = self.engine.process_text(text.strip(), src=lang, speak=False)
        self.show(r, "Typed")
        if self.cfg.talk.speak_to_me:
            self.speak(r.tgt_text, r.tgt_lang, f"{self.my_name} translation")
        return "reply"

    def translate_file(self) -> str:
        raw = self.keys.line_input("  audio file (drag it here)> ")
        if not raw or not raw.strip():
            return "ready"
        path = Path(raw.strip().strip("'\"")).expanduser()
        if not path.is_file():
            self.say(f"  no such file: {path}", "red")
            return "ready"
        try:
            audio = read_audio(path, SAMPLE_RATE)
        except Exception as e:  # noqa: BLE001 - unreadable/unsupported file
            self.say(f"  cannot read {path.name}: {e}", "red")
            return "ready"
        seg = Segmenter(make_vad(self.cfg.vad), self.cfg.vad)
        utts = seg.feed(audio)
        last = seg.flush()
        if last is not None:
            utts.append(last)
        self.say(f"  {path.name}: {audio.size / SAMPLE_RATE:.0f}s, {len(utts)} part(s)", "dim")
        clips = []
        for i, utt in enumerate(utts, 1):
            with self.console.status(f"Translating part {i}/{len(utts)}…"):
                r = self.engine.process_audio(utt.audio, speak=self.cfg.talk.speak_to_me)
            if r is not None and r.tgt_text:
                self.show(r, f"{path.name} · part {i}" if len(utts) > 1 else path.name)
                clips.append((r.audio, r.sample_rate))
        if not clips:
            self.say("  no speech found in the file", "yellow")
            return "ready"
        audio_out, sr = concat_with_gaps(clips)
        if audio_out.size:
            self.play(audio_out, sr, "translation")
        return "ready"

    def replay(self) -> None:
        if self.last_audio is None:
            self.say("  nothing to replay yet", "dim")
            return
        self.play(*self.last_audio, "replay")

    def toggle(self, key: str, label: str) -> None:
        value = not getattr(self.cfg.talk, key)
        self._remember({"talk": {key: value}})
        setattr(self.cfg.talk, key, value)
        self.say(Text.assemble(("  ", ""), (label, "bold"), (": ", ""), ("on" if value else "off",
                                                                         "bold green" if value else "bold red")))

    def _remember(self, updates: dict) -> None:
        if not self.save_choices:
            return
        try:
            save_local(self.cfg, updates)
        except OSError as e:
            log.warning("could not save your choice: %s", e)

    def voices_menu(self) -> None:
        from .tts import list_voices

        tts = self.engine.tts
        if tts is None:
            return
        options = {lang: list_voices(self.cfg, lang) for lang in (self.their, self.mine)}
        changed = False
        while True:
            t = Table(box=None, show_header=True, padding=(0, 2))
            t.add_column("", style="bold reverse", justify="center")
            t.add_column("language")
            t.add_column("voice")
            t.add_column("choices", style="dim")
            for key, lang in (("I", self.their), ("E", self.mine)):
                current = next((v for v in options[lang] if v.id == tts.voices[lang]), None)
                t.add_row(key, get_language(lang).name, current.label if current else tts.voices[lang],
                          f"{len(options[lang])} installed")
            self.console.print(Panel(t, title="voices", title_align="left", border_style="blue"))
            self.hint(("I", f"next {self.their_name} voice"), ("E", f"next {self.my_name} voice"),
                      ("P", "play both"), ("ESC", "done"))
            key = self.keys.get()
            if key in ("i", "e"):
                lang = self.their if key == "i" else self.mine
                ids = [v.id for v in options[lang]]
                if not ids:
                    self.say(f"  no {get_language(lang).name} voices installed", "yellow")
                    continue
                cur = ids.index(tts.voices[lang]) if tts.voices[lang] in ids else -1
                new = ids[(cur + 1) % len(ids)]
                try:
                    with self.console.status("Loading voice…"):
                        tts.set_voice(lang, new)
                        audio, sr = tts.synthesize(SAMPLE_SENTENCES.get(lang, "Ok."), lang)
                except Exception as e:  # noqa: BLE001 - broken/missing voice file
                    self.say(f"  cannot use {new}: {e}", "red")
                    continue
                changed = True
                self.play(audio, sr, f"{get_language(lang).name} voice sample")
            elif key == "p":
                for lang in (self.their, self.mine):
                    audio, sr = tts.synthesize(SAMPLE_SENTENCES.get(lang, "Ok."), lang)
                    self.play(audio, sr, f"{get_language(lang).name} voice sample")
            elif key in ("esc", "q", "enter", "v", "s"):
                break
        if changed:
            self._remember({"tts": {"voices": {self.their: tts.voices[self.their], self.mine: tts.voices[self.mine]}}})
            self.cfg.tts.voices.update(tts.voices)
            self.say("  voices saved", "dim")
        self.say(self.status_line())

    # -- main loop
    def run(self) -> None:
        self.banner()
        state, show_hint = "ready", True
        while state != "quit":
            if show_hint:
                if state == "ready":
                    self.hint(("SPACE", f"listen to {self.their_name}"), ("E", f"speak {self.my_name}"),
                              ("T", "type"), ("H", "help"), ("Q", "quit"))
                else:
                    self.say(Text.assemble(("Your turn ", f"bold {MY_COLOR}"),
                                           (f"— answer in {self.my_name}", MY_COLOR)))
                    self.hint(("SPACE", f"speak {self.my_name}"), ("T", "type"), ("L", "listen to them"),
                              ("R", "replay"), ("S", "skip"))
            key = self.keys.get()
            show_hint = key in KNOWN_KEYS
            if not show_hint:
                continue
            try:
                state = self.step(state, key)
            except Exception as e:  # noqa: BLE001 - keep the conversation going
                log.debug("step failed", exc_info=True)
                self.say(f"  something went wrong: {type(e).__name__}: {e}  (run with -v for details)", "red")
                state = "ready"

    def step(self, state: str, key: str | None) -> str:
        """Handle one key in ``state`` ("ready" | "reply"); returns the next state."""
        if key in ("q", "ctrl-d"):
            return "quit"
        if key in ("space", "enter"):
            return self.their_turn() if state == "ready" else self.my_turn()
        actions: dict[str, Callable[[], str]] = {
            "e": self.my_turn,
            "l": self.their_turn,
            "t": lambda: self.type_text(self.mine),
            "i": lambda: self.type_text(self.their),
            "f": self.translate_file,
        }
        if key in actions:
            return actions[key]()
        if key in ("s", "esc"):
            return "ready"
        if key == "r":
            self.replay()
        elif key == "a":
            self.toggle("speak_to_me", f"{self.my_name} audio")
        elif key == "c":
            self.toggle("confirm_my_speech", "confirm my speech")
        elif key == "v":
            self.voices_menu()
        elif key in ("h", "?"):
            self.help()
        return state


def run_talk(engine: Engine, cfg: Config, console: Console) -> None:
    with TerminalKeys() as keys:
        app = TalkApp(engine, cfg, keys, console=console)
        try:
            app.run()
        except KeyboardInterrupt:
            pass
        finally:
            app.player.stop()
    console.print("bye 👋")
