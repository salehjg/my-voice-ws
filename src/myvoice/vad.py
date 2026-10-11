"""Voice activity detection and utterance segmentation.

``Segmenter`` turns a continuous 16 kHz stream into utterances: it opens when the VAD
probability crosses ``threshold``, keeps ``pre_roll_ms`` of audio from before the onset, and
closes after ``min_silence_ms`` of silence (or at ``max_utterance_s``).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import VadConfig

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
TAIL_KEEP_MS = 200  # trailing silence kept on each utterance (Whisper likes a little context)


class FrameVad(Protocol):
    frame_size: int

    def prob(self, frame: np.ndarray) -> float: ...

    def reset(self) -> None: ...


class SileroVad:
    """Silero VAD (the model ships inside the ``silero-vad`` pip package; runs on CPU)."""

    frame_size = 512  # Silero v5+ requires exactly 512 samples per call at 16 kHz

    def __init__(self) -> None:
        import torch
        from silero_vad import load_silero_vad

        torch.set_num_threads(max(1, min(torch.get_num_threads(), 4)))
        self._torch = torch
        self.model = load_silero_vad()

    def prob(self, frame: np.ndarray) -> float:
        with self._torch.inference_mode():
            return float(self.model(self._torch.from_numpy(frame), SAMPLE_RATE).item())

    def reset(self) -> None:
        self.model.reset_states()


class EnergyVad:
    """RMS-energy VAD: a dependency-free fallback (fine in a quiet room, poor in noise)."""

    frame_size = 512

    def __init__(self, threshold_db: float = -42.0) -> None:
        self.threshold_db = threshold_db

    def prob(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)) + 1e-10)
        db = 20.0 * math.log10(rms)
        return 1.0 / (1.0 + math.exp(-(db - self.threshold_db) / 3.0))

    def reset(self) -> None:
        pass


def make_vad(cfg: VadConfig) -> FrameVad:
    if cfg.backend == "silero":
        try:
            return SileroVad()
        except ImportError as e:
            log.warning("silero-vad unavailable (%s); falling back to the energy VAD", e)
    return EnergyVad(cfg.energy_threshold_db)


@dataclass
class Utterance:
    audio: np.ndarray  # float32 mono, 16 kHz
    start_s: float
    end_s: float
    forced_cut: bool = False

    @property
    def duration_s(self) -> float:
        return self.audio.size / SAMPLE_RATE


class Segmenter:
    def __init__(self, vad: FrameVad, cfg: VadConfig) -> None:
        self.vad = vad
        self.cfg = cfg
        self.frame = vad.frame_size
        frame_ms = self.frame * 1000.0 / SAMPLE_RATE
        self.threshold = cfg.threshold
        self.neg_threshold = max(cfg.threshold - 0.15, 0.01)
        self.pre_roll_frames = max(0, math.ceil(cfg.pre_roll_ms / frame_ms))
        self.min_speech_frames = max(1, math.ceil(cfg.min_speech_ms / frame_ms))
        self.min_silence_frames = max(1, math.ceil(cfg.min_silence_ms / frame_ms))
        self.tail_keep_frames = min(self.min_silence_frames, math.ceil(TAIL_KEEP_MS / frame_ms))
        self.max_frames = max(self.min_speech_frames + 1, int(cfg.max_utterance_s * 1000 / frame_ms))
        self._pending = np.zeros(0, dtype=np.float32)
        self._frames_seen = 0
        self._clear()

    def _clear(self) -> None:
        self._preroll: deque[np.ndarray] = deque(maxlen=self.pre_roll_frames)
        self._frames: list[np.ndarray] = []
        self._triggered = False
        self._speech = 0
        self._silence = 0
        self._start_frame = 0

    @property
    def in_speech(self) -> bool:
        return self._triggered

    def reset(self) -> None:
        """Drop everything buffered (used while the mic is muted during playback)."""
        self._pending = np.zeros(0, dtype=np.float32)
        self._clear()
        self.vad.reset()

    def feed(self, samples: np.ndarray) -> list[Utterance]:
        buf = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        n = buf.size // self.frame
        out: list[Utterance] = []
        for i in range(n):
            utt = self._step(buf[i * self.frame : (i + 1) * self.frame])
            if utt is not None:
                out.append(utt)
        self._pending = buf[n * self.frame :].copy()
        return out

    def flush(self) -> Utterance | None:
        """End of stream: emit the utterance in progress, if it is long enough."""
        utt = self._finish(forced=False) if self._triggered else None
        self.reset()
        return utt

    def _step(self, frame: np.ndarray) -> Utterance | None:
        p = self.vad.prob(frame)
        self._frames_seen += 1
        if not self._triggered:
            if p >= self.threshold:
                self._triggered = True
                self._frames = [*self._preroll, frame]
                self._start_frame = self._frames_seen - len(self._frames)
                self._speech, self._silence = 1, 0
            else:
                self._preroll.append(frame)
            return None

        self._frames.append(frame)
        if p >= self.threshold:
            self._speech += 1
            self._silence = 0
        elif p < self.neg_threshold or self._silence:
            # Below neg_threshold starts the silence timer; the hysteresis band only extends it.
            self._silence += 1

        if self._silence >= self.min_silence_frames:
            return self._finish(forced=False)
        if len(self._frames) >= self.max_frames:
            return self._finish(forced=True)
        return None

    def _finish(self, forced: bool) -> Utterance | None:
        frames = self._frames
        if not forced and self._silence > self.tail_keep_frames:
            frames = frames[: len(frames) - (self._silence - self.tail_keep_frames)]
        enough = self._speech >= self.min_speech_frames
        start = self._start_frame * self.frame / SAMPLE_RATE
        utt = None
        if enough and frames:
            audio = np.concatenate(frames)
            utt = Utterance(audio=audio, start_s=start, end_s=start + audio.size / SAMPLE_RATE, forced_cut=forced)
        self._clear()
        self.vad.reset()
        return utt
