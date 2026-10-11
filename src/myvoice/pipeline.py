"""The speech-to-speech pipeline: ASR (Whisper) -> MT (LLM) -> TTS (Kokoro/Piper), plus file translation."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from .audio import concat_with_gaps, read_audio, resample, write_wav
from .config import Config
from .langs import guess_text_language, other_language, parse_direction
from .vad import SAMPLE_RATE, Segmenter, make_vad

if TYPE_CHECKING:
    from .asr import Transcript, WhisperAsr
    from .translate import Translator
    from .tts import Tts

log = logging.getLogger(__name__)


@dataclass
class TurnResult:
    src_lang: str
    tgt_lang: str
    src_text: str
    tgt_text: str
    lang_prob: float = 1.0
    utterance_s: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)
    audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32), repr=False)
    sample_rate: int = 22050

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("audio")
        d["time"] = datetime.now().isoformat(timespec="seconds")
        return d


def gpu_memory_mib() -> tuple[int, int] | None:
    """(used, total) MiB on GPU 0 as reported by nvidia-smi, or None."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        used, total = (int(x) for x in out.split(","))
        return used, total
    except Exception:  # noqa: BLE001 - informational only
        return None


def format_turn(r: TurnResult) -> str:
    t = r.timings
    timing = " · ".join(f"{k} {v:.2f}s" for k, v in t.items())
    conf = f" p={r.lang_prob:.2f}" if r.lang_prob < 0.999 else ""
    return (
        f"\n[{r.src_lang}→{r.tgt_lang}{conf}] {r.src_text}\n"
        f"        → {r.tgt_text}\n"
        f"        ({timing})"
    )


class Engine:
    """Holds the loaded models and runs one turn (utterance or text) through them."""

    def __init__(self, cfg: Config, asr: WhisperAsr | None, mt: Translator, tts: Tts | None) -> None:
        self.cfg = cfg
        self.asr = asr
        self.mt = mt
        self.tts = tts
        self.languages = list(cfg.general.languages)
        self.fixed = parse_direction(cfg.general.direction, self.languages)

    @classmethod
    def load(cls, cfg: Config, need_asr: bool = True, need_tts: bool = True) -> Engine:
        from .translate import make_translator

        asr = tts = None
        if need_asr:
            from .asr import WhisperAsr

            t0 = time.perf_counter()
            asr = WhisperAsr(cfg.asr, cfg.path(cfg.asr.model))
            log.info("ASR loaded in %.1fs (%s, %s)", time.perf_counter() - t0, cfg.asr.model, cfg.asr.compute_type)
        t0 = time.perf_counter()
        mt = make_translator(cfg)
        log.info("translator loaded in %.1fs (%s: %s)", time.perf_counter() - t0, cfg.translation.backend, cfg.translation.model)
        if need_tts:
            from .tts import make_tts

            t0 = time.perf_counter()
            tts = make_tts(cfg)
            log.info("TTS loaded in %.1fs", time.perf_counter() - t0)
        mem = gpu_memory_mib()
        if mem:
            log.info("GPU memory in use: %d / %d MiB", *mem)
        return cls(cfg, asr, mt, tts)

    def warmup(self) -> None:
        t0 = time.perf_counter()
        a, b = self.languages
        if self.asr is not None:
            noise = (np.random.default_rng(0).standard_normal(SAMPLE_RATE) * 0.01).astype(np.float32)
            self.asr.transcribe(noise, self.languages, forced_language=a)
        self.mt.translate("Ciao." if a == "it" else "Hello.", a, b)
        self.mt.history.clear()
        if self.tts is not None:
            self.tts.synthesize("Ok.", b)
        log.info("warm-up done in %.1fs", time.perf_counter() - t0)

    def _directions(self, detected: str | None) -> tuple[str, str]:
        if self.fixed:
            return self.fixed
        assert detected is not None
        return detected, other_language(detected, self.languages)

    def transcribe(self, audio: np.ndarray, lang: str | None = None) -> Transcript | None:
        """Speech -> text, in ``lang`` if given (no language detection), else detected."""
        assert self.asr is not None, "engine was loaded without ASR"
        if lang is None and self.fixed:
            lang = self.fixed[0]
        return self.asr.transcribe(audio, self.languages, forced_language=lang)

    def _finish(self, r: TurnResult, speak: bool) -> TurnResult:
        if not r.tgt_text:
            return r
        t0 = time.perf_counter()
        r.tgt_text = r.tgt_text.strip()
        if speak and self.tts is not None:
            r.audio, r.sample_rate = self.tts.synthesize(r.tgt_text, r.tgt_lang)
            r.timings["tts"] = time.perf_counter() - t0
        return r

    def process_audio(self, audio: np.ndarray, speak: bool = True, src: str | None = None) -> TurnResult | None:
        """One utterance (float32 mono 16 kHz) -> transcript, translation and speech.

        ``src`` fixes the spoken language (no detection); otherwise ``general.direction`` decides.
        """
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        tr = self.transcribe(audio, src)
        timings["asr"] = time.perf_counter() - t0
        if tr is None:
            log.debug("no speech recognised in %.1fs utterance", audio.size / SAMPLE_RATE)
            return None
        if src is not None:
            src, tgt = src, other_language(src, self.languages)
        else:
            src, tgt = self._directions(tr.language)
        t0 = time.perf_counter()
        translation = self.mt.translate(tr.text, src, tgt)
        timings["mt"] = time.perf_counter() - t0
        r = TurnResult(src, tgt, tr.text, translation, tr.language_prob, audio.size / SAMPLE_RATE, timings)
        return self._finish(r, speak)

    def process_text(self, text: str, src: str | None = None, speak: bool = True) -> TurnResult:
        """Text -> translation (+ speech). An explicit ``src`` wins over ``general.direction``."""
        if src is None and self.fixed:
            src, tgt = self.fixed
        else:
            src = src or guess_text_language(text, self.languages)
            tgt = other_language(src, self.languages)
        t0 = time.perf_counter()
        translation = self.mt.translate(text, src, tgt)
        r = TurnResult(src, tgt, text, translation, timings={"mt": time.perf_counter() - t0})
        return self._finish(r, speak)


class SessionLog:
    def __init__(self, cfg: Config, prefix: str = "session") -> None:
        self.path: Path | None = None
        if cfg.general.session_log_dir:
            d = cfg.path(cfg.general.session_log_dir)
            d.mkdir(parents=True, exist_ok=True)
            self.path = d / f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}.jsonl"

    def write(self, r: TurnResult) -> None:
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")


def translate_file(
    engine: Engine, cfg: Config, in_path: Path, out_path: Path | None, speak: bool = False,
    printer: Callable[[str], None] = print,
) -> list[TurnResult]:
    """Offline: segment a recording with the VAD, translate every utterance, optionally write one WAV."""
    audio = read_audio(in_path, SAMPLE_RATE)
    seg = Segmenter(make_vad(cfg.vad), cfg.vad)
    utts = seg.feed(audio)
    last = seg.flush()
    if last is not None:
        utts.append(last)
    log.info("%s: %.1fs audio, %d utterances", in_path.name, audio.size / SAMPLE_RATE, len(utts))
    results: list[TurnResult] = []
    for utt in utts:
        r = engine.process_audio(utt.audio, speak=speak or out_path is not None)
        if r is None:
            continue
        printer(f"[{utt.start_s:7.2f}s]" + format_turn(r))
        results.append(r)
    if out_path is not None:
        track, sr = concat_with_gaps([(r.audio, r.sample_rate) for r in results])
        write_wav(out_path, track, sr)
        with open(out_path.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
        printer(f"\nwrote {out_path} and {out_path.with_suffix('.jsonl')}")
    return results


SELFTEST_SENTENCES = {
    "it": "Buongiorno! Vorrei prenotare un tavolo per due persone stasera alle otto, se possibile vicino alla finestra.",
    "en": "Excuse me, could you tell me how to get to the train station? I have to catch the five o'clock train.",
    "es": "Buenos días, ¿podría decirme dónde está la estación de tren?",
    "fr": "Bonjour, pourriez-vous me dire où se trouve la gare, s'il vous plaît ?",
    "de": "Guten Tag, können Sie mir sagen, wo der Bahnhof ist?",
    "pt": "Bom dia, pode dizer-me onde fica a estação de comboios?",
}


def selftest(engine: Engine, cfg: Config, out_dir: Path) -> bool:
    """No microphone needed: a voice speaks a sentence, then it runs through the whole pipeline."""
    assert engine.tts is not None and engine.asr is not None
    ok = True
    for src in engine.languages:
        if engine.fixed and engine.fixed[0] != src:
            continue
        sentence = SELFTEST_SENTENCES[src]
        speech, sr = engine.tts.synthesize(sentence, src)
        speech16 = resample(speech, sr, SAMPLE_RATE)
        # pad with a little silence like a real VAD segment
        pad = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)
        speech16 = np.concatenate([pad, speech16, pad])
        t0 = time.perf_counter()
        r = engine.process_audio(speech16)
        total = time.perf_counter() - t0
        print(f"\n=== self-test {src} ===\ninput : {sentence}")
        if r is None:
            print("FAILED: nothing recognised")
            ok = False
            continue
        print(format_turn(r))
        print(f"        end-to-end {total:.2f}s for {speech16.size / SAMPLE_RATE:.1f}s of speech")
        if r.src_lang != src:
            print(f"WARNING: language detected as {r.src_lang!r}, expected {src!r}")
            ok = False
        write_wav(out_dir / f"selftest_{src}_input.wav", speech, sr)
        if r.audio.size:
            write_wav(out_dir / f"selftest_{src}_to_{r.tgt_lang}.wav", r.audio, r.sample_rate)
    mem = gpu_memory_mib()
    if mem:
        print(f"\nGPU memory in use: {mem[0]} / {mem[1]} MiB")
    print(f"audio written to {out_dir}/selftest_*.wav")
    return ok
