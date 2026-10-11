"""Text-to-speech: Piper and Kokoro voices, one chosen per language.

A voice is named by an id:

* Piper: the voice file name without ``.onnx``, e.g. ``it_IT-paola-medium`` (looked up under
  ``tts.piper_dir``); a path to an ``.onnx`` file also works.
* Kokoro: ``kokoro:<name>``, e.g. ``kokoro:im_nicola`` (needs ``tts.kokoro_model`` +
  ``tts.kokoro_voices``).

Engines are loaded on first use, so unused voices cost nothing.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config

log = logging.getLogger(__name__)

KOKORO_PREFIX = "kokoro:"

# Kokoro voice names start with a language letter (a = US English, b = British English,
# i = Italian, ...) and a gender letter. Only languages we translate are listed.
KOKORO_LANG = {"a": ("en", "en-us"), "b": ("en", "en-gb"), "i": ("it", "it"), "e": ("es", "es"),
               "f": ("fr", "fr-fr"), "p": ("pt", "pt-br")}

# Voices offered in the voice menu (Kokoro ships many more; any of them works in config.toml).
KOKORO_RECOMMENDED = ["if_sara", "im_nicola", "af_heart", "af_bella", "am_michael", "am_fenrir",
                      "bf_emma", "bm_george"]

# Piper's .onnx.json has no gender, so the known ones are listed here for nicer labels.
PIPER_GENDER = {"paola": "female", "serena": "female", "riccardo": "male", "lessac": "female",
                "ryan": "male", "cori": "female", "alan": "male", "amy": "female", "joe": "male",
                "kristin": "female", "john": "male", "norman": "male", "ljspeech": "female",
                "jenny_dioco": "female", "alba": "female", "northern_english_male": "male",
                "southern_english_female": "female", "bryce": "male", "danny": "male",
                "hfc_female": "female", "hfc_male": "male", "kathleen": "female", "kusal": "male",
                "mike": "male", "sam": "male", "reza_ibrahim": "male"}

REGION = {"en_US": "US", "en_GB": "UK", "en-us": "US", "en-gb": "UK"}


@dataclass(frozen=True)
class VoiceInfo:
    id: str
    engine: str  # "piper" | "kokoro"
    lang: str  # "it", "en", ...
    name: str  # "Paola"
    detail: str  # "female · medium" / "male · US"

    @property
    def label(self) -> str:
        return f"{self.name} ({self.detail}, {self.engine.capitalize()})"


# --------------------------------------------------------------------------- catalogue


def _piper_info(onnx: Path) -> VoiceInfo | None:
    meta_path = Path(f"{onnx}.json")
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        lang = meta["language"]["family"]
        region = meta["language"].get("code", "")
    except (OSError, ValueError, KeyError):
        return None
    # it_IT-paola-medium -> ("paola", "medium")
    parts = onnx.stem.split("-")
    speaker = parts[1] if len(parts) >= 3 else onnx.stem
    quality = parts[-1] if len(parts) >= 3 else ""
    bits = [b for b in (PIPER_GENDER.get(speaker, ""), REGION.get(region, ""), quality.replace("_", "-")) if b]
    return VoiceInfo(onnx.stem, "piper", lang, speaker.replace("_", " ").title(), " · ".join(bits))


def _kokoro_info(name: str) -> VoiceInfo | None:
    if len(name) < 3 or name[0] not in KOKORO_LANG or name[2] != "_":
        return None
    lang, espeak = KOKORO_LANG[name[0]]
    gender = {"f": "female", "m": "male"}.get(name[1], "")
    bits = [b for b in (gender, REGION.get(espeak, "")) if b]
    return VoiceInfo(KOKORO_PREFIX + name, "kokoro", lang, name[3:].replace("_", " ").title(), " · ".join(bits))


def kokoro_available(cfg: Config) -> bool:
    return cfg.path(cfg.tts.kokoro_model).is_file() and cfg.path(cfg.tts.kokoro_voices).is_file()


def list_voices(cfg: Config, lang: str | None = None, all_kokoro: bool = False) -> list[VoiceInfo]:
    """Installed voices (optionally for one language): Piper voices first, then Kokoro."""
    out: list[VoiceInfo] = []
    piper_dir = cfg.path(cfg.tts.piper_dir)
    if piper_dir.is_dir():
        for onnx in sorted(piper_dir.rglob("*.onnx")):
            info = _piper_info(onnx)
            if info is not None:
                out.append(info)
    if kokoro_available(cfg):
        try:
            with np.load(cfg.path(cfg.tts.kokoro_voices)) as pack:
                names = sorted(pack.files)
        except (OSError, ValueError) as e:
            log.warning("cannot read Kokoro voices %s: %s", cfg.tts.kokoro_voices, e)
            names = []
        if not all_kokoro:
            names = [n for n in KOKORO_RECOMMENDED if n in names]
        out.extend(info for n in names if (info := _kokoro_info(n)) is not None)
    if lang is not None:
        out = [v for v in out if v.lang == lang]
    return out


def resolve_piper(cfg: Config, voice_id: str) -> Path:
    if voice_id.endswith(".onnx"):
        return cfg.path(voice_id)
    matches = sorted(cfg.path(cfg.tts.piper_dir).rglob(f"{voice_id}.onnx"))
    if not matches:
        raise FileNotFoundError(f"Piper voice {voice_id!r} not found under {cfg.tts.piper_dir}; see to-download.md")
    return matches[0]


def voice_language(cfg: Config, voice_id: str) -> str | None:
    """Language a voice speaks (None if unknown)."""
    if voice_id.startswith(KOKORO_PREFIX):
        info = _kokoro_info(voice_id[len(KOKORO_PREFIX):])
        return info.lang if info else None
    try:
        info = _piper_info(resolve_piper(cfg, voice_id))
    except FileNotFoundError:
        return None
    return info.lang if info else None


# --------------------------------------------------------------------------- engines


class _PiperEngine:
    def __init__(self, cfg: Config, onnx: Path) -> None:
        from piper import PiperVoice, SynthesisConfig

        if not onnx.is_file() or not Path(f"{onnx}.json").is_file():
            raise FileNotFoundError(f"Piper voice not found: {onnx} (+ .json); see to-download.md")
        self.cfg = cfg.tts
        self.voice = PiperVoice.load(str(onnx), use_cuda=cfg.tts.use_cuda)
        self.sample_rate = int(self.voice.config.sample_rate)
        self.syn_config = SynthesisConfig(
            length_scale=cfg.tts.length_scale if cfg.tts.length_scale > 0 else None,
            volume=cfg.tts.volume,
        )

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        sr = self.sample_rate
        gap = np.zeros(int(sr * self.cfg.sentence_silence_ms / 1000), dtype=np.float32)
        parts: list[np.ndarray] = []
        for chunk in self.voice.synthesize(text, syn_config=self.syn_config):  # one chunk per sentence
            if parts:
                parts.append(gap)
            parts.append(np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1))
        if not parts:
            return np.zeros(0, dtype=np.float32), sr
        return np.concatenate(parts), sr


class _KokoroEngine:
    def __init__(self, cfg: Config) -> None:
        from kokoro_onnx import Kokoro

        model, voices = cfg.path(cfg.tts.kokoro_model), cfg.path(cfg.tts.kokoro_voices)
        if not model.is_file() or not voices.is_file():
            raise FileNotFoundError(f"Kokoro model not found ({model}, {voices}); see to-download.md")
        self.cfg = cfg.tts
        self.kokoro = Kokoro(str(model), str(voices))

    def synthesize(self, text: str, name: str) -> tuple[np.ndarray, int]:
        info = _kokoro_info(name)
        if info is None:
            raise ValueError(f"unknown Kokoro voice {name!r}")
        espeak_lang = KOKORO_LANG[name[0]][1]
        speed = 1.0 / self.cfg.length_scale if self.cfg.length_scale > 0 else 1.0
        audio, sr = self.kokoro.create(
            text, voice=name, speed=speed, lang=espeak_lang, sentence_pause=self.cfg.sentence_silence_ms / 1000
        )
        audio = np.asarray(audio, dtype=np.float32).reshape(-1) * self.cfg.volume
        return audio, int(sr)


class Tts:
    """Speaks text in a language with the voice currently chosen for it."""

    def __init__(self, cfg: Config, voices: dict[str, str] | None = None) -> None:
        self.cfg = cfg
        self.voices = dict(voices if voices is not None else cfg.tts.voices)
        self._piper: dict[Path, _PiperEngine] = {}
        self._kokoro: _KokoroEngine | None = None
        self._lock = threading.Lock()  # engines are not re-entrant
        for lang in cfg.general.languages:
            self._engine_for(self.voices[lang])  # fail early on a missing voice

    def _engine_for(self, voice_id: str):
        if voice_id.startswith(KOKORO_PREFIX):
            if self._kokoro is None:
                self._kokoro = _KokoroEngine(self.cfg)
            return self._kokoro
        path = resolve_piper(self.cfg, voice_id)
        if path not in self._piper:
            self._piper[path] = _PiperEngine(self.cfg, path)
        return self._piper[path]

    def set_voice(self, lang: str, voice_id: str) -> None:
        with self._lock:
            self._engine_for(voice_id)
        self.voices[lang] = voice_id

    def synthesize(self, text: str, lang: str, voice_id: str | None = None) -> tuple[np.ndarray, int]:
        """Return (float32 mono audio, sample rate); empty audio for text with nothing to say."""
        voice_id = voice_id or self.voices[lang]
        with self._lock:
            engine = self._engine_for(voice_id)
            if not any(ch.isalnum() for ch in text):  # "…", "?!", emoji: Kokoro would raise
                return np.zeros(0, dtype=np.float32), 24000 if isinstance(engine, _KokoroEngine) else engine.sample_rate
            if isinstance(engine, _KokoroEngine):
                return engine.synthesize(text, voice_id[len(KOKORO_PREFIX):])
            return engine.synthesize(text)


def make_tts(cfg: Config) -> Tts:
    return Tts(cfg)
