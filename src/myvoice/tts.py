"""Text-to-speech with Piper (ONNX VITS voices, one per language)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .config import Config, TtsConfig

log = logging.getLogger(__name__)


class PiperTts:
    def __init__(self, cfg: TtsConfig, voice_paths: dict[str, Path]) -> None:
        from piper import PiperVoice, SynthesisConfig

        self.cfg = cfg
        self.voices = {}
        for lang, path in voice_paths.items():
            if not path.is_file() or not Path(f"{path}.json").is_file():
                raise FileNotFoundError(f"Piper voice for {lang!r} not found: {path} (+ .json); see to-download.md")
            self.voices[lang] = PiperVoice.load(str(path), use_cuda=cfg.use_cuda)
        self.syn_config = SynthesisConfig(
            length_scale=cfg.length_scale if cfg.length_scale > 0 else None,
            volume=cfg.volume,
        )

    def sample_rate(self, lang: str) -> int:
        return int(self.voices[lang].config.sample_rate)

    def synthesize(self, text: str, lang: str) -> tuple[np.ndarray, int]:
        """Return (float32 mono audio, sample rate). Piper yields one chunk per sentence."""
        voice = self.voices[lang]
        sr = self.sample_rate(lang)
        gap = np.zeros(int(sr * self.cfg.sentence_silence_ms / 1000), dtype=np.float32)
        parts: list[np.ndarray] = []
        for chunk in voice.synthesize(text, syn_config=self.syn_config):
            if parts:
                parts.append(gap)
            parts.append(np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1))
        if not parts:
            return np.zeros(0, dtype=np.float32), sr
        return np.concatenate(parts), sr


def make_tts(cfg: Config) -> PiperTts:
    voices = {lang: cfg.path(cfg.tts.voices[lang]) for lang in cfg.general.languages}
    return PiperTts(cfg.tts, voices)
