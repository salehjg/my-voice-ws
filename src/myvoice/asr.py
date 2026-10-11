"""Speech recognition with faster-whisper (CTranslate2 on CUDA).

Language identification is restricted to the configured pair: Whisper scores all ~100
languages, and short Italian clips are sometimes labelled Spanish/Portuguese. We take the
most likely language *of the pair* and, if Whisper had picked something else, re-decode
with that language forced.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import AsrConfig

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    text: str
    language: str
    language_prob: float  # renormalised over the language pair (1.0 when forced)
    avg_logprob: float
    no_speech_prob: float


# Phrases Whisper is known to "hear" in silence/noise (learned from subtitle credits).
_HALLUCINATION_SUBSTRINGS = (
    "amara.org",
    "sottotitoli creati dalla comunità",
    "sottotitoli a cura di",
    "sottotitoli e revisione a cura di",
    "subtitles by the amara",
    "qtss",
)
_HALLUCINATION_EXACT = {
    "thank you for watching",
    "thanks for watching",
    "thank you for watching and see you next time",
    "please subscribe",
    "subscribe to the channel",
    "grazie per la visione",
    "grazie a tutti per la visione",
    "iscriviti al canale",
    "you",
}


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = re.sub(r"[\s]+", " ", text)
    return text.strip(" .!?…,;:-–—\"'«»“”")


def is_hallucination(text: str) -> bool:
    norm = _normalise(text)
    if not norm or not re.search(r"\w", norm):
        return True
    if any(s in norm for s in _HALLUCINATION_SUBSTRINGS):
        return True
    return norm in _HALLUCINATION_EXACT


def choose_language(
    all_probs: Sequence[tuple[str, float]] | None,
    detected: str,
    detected_prob: float,
    allowed: Sequence[str],
) -> tuple[str, float]:
    """Pick the most likely language among ``allowed``; confidence renormalised over them."""
    probs = dict(all_probs) if all_probs else {detected: detected_prob}
    cand = {lang: float(probs.get(lang, 0.0)) for lang in allowed}
    best = max(allowed, key=lambda lang: cand[lang])
    total = sum(cand.values())
    conf = cand[best] / total if total > 0 else 1.0 / len(allowed)
    return best, conf


class WhisperAsr:
    def __init__(self, cfg: AsrConfig, model_path: str | Path) -> None:
        from faster_whisper import WhisperModel

        self.cfg = cfg
        model_path = Path(model_path)
        if not (model_path / "model.bin").exists():
            raise FileNotFoundError(f"Whisper model not found in {model_path} (see to-download.md)")
        self.model = WhisperModel(
            str(model_path),
            device=cfg.device,
            compute_type=cfg.compute_type,
            local_files_only=True,
        )

    def _run(self, audio: np.ndarray, language: str | None):
        return self.model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=self.cfg.beam_size,
            temperature=[0.0, 0.2, 0.4, 0.6],
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,  # utterances are already segmented by our VAD
            initial_prompt=self.cfg.initial_prompt or None,
        )

    def transcribe(
        self, audio: np.ndarray, allowed: Sequence[str], forced_language: str | None = None
    ) -> Transcript | None:
        """Transcribe one utterance (float32, 16 kHz). Returns ``None`` if nothing usable was said."""
        segments, info = self._run(audio, forced_language)
        if forced_language is None:
            language, conf = choose_language(
                info.all_language_probs, info.language, info.language_probability, allowed
            )
            if info.language != language:
                log.debug("Whisper detected %r (p=%.2f); re-decoding as %r", info.language, info.language_probability, language)
                segments, info = self._run(audio, language)
        else:
            language, conf = forced_language, 1.0

        segs = list(segments)  # decoding happens here (segments is a generator)
        text = " ".join(s.text.strip() for s in segs).strip()
        if not segs or not text:
            return None
        if self.cfg.filter_hallucinations and is_hallucination(text):
            log.info("dropped likely Whisper hallucination: %r", text)
            return None
        return Transcript(
            text=text,
            language=language,
            language_prob=conf,
            avg_logprob=float(np.mean([s.avg_logprob for s in segs])),
            no_speech_prob=float(max(s.no_speech_prob for s in segs)),
        )
