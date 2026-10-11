"""Audio helpers: resampling, WAV I/O, device lookup, microphone capture and playback."""

from __future__ import annotations

import logging
import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False)


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Band-limited resampling (scipy polyphase) with a linear-interpolation fallback."""
    if sr_from == sr_to or audio.size == 0:
        return audio.astype(np.float32, copy=False)
    try:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr_from, sr_to)
        return resample_poly(audio, sr_to // g, sr_from // g).astype(np.float32)
    except ImportError:
        n_out = int(round(audio.size * sr_to / sr_from))
        x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, audio).astype(np.float32)


def float_to_int16(audio: np.ndarray) -> np.ndarray:
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(float_to_int16(to_mono_float32(audio)).tobytes())
    return path


def _decode_with_pyav(path: Path, sample_rate: int) -> np.ndarray:
    # Same approach as faster_whisper.decode_audio, which in 1.2.1 passes an argument
    # (`metadata_errors`) that PyAV >= 16 no longer accepts.
    import av

    resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
    chunks: list[np.ndarray] = []
    with av.open(str(path), mode="r") as container:
        frames = container.decode(audio=0)
        while True:
            try:
                frame = next(frames)
            except StopIteration:
                break
            except av.error.InvalidDataError:  # skip corrupt frames, like faster-whisper
                continue
            chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        chunks.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))  # flush
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def read_audio(path: str | Path, sample_rate: int = 16000) -> np.ndarray:
    """Read any audio file as mono float32 at ``sample_rate``.

    Uses PyAV (installed with faster-whisper; handles wav/mp3/flac/ogg/m4a/...); falls back to
    the stdlib ``wave`` module for plain 16-bit PCM WAV files.
    """
    path = Path(path)
    try:
        return _decode_with_pyav(path, sample_rate)
    except ImportError:
        pass
    with wave.open(str(path), "rb") as wf:
        if wf.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported without faster-whisper installed")
        frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        frames = frames.reshape(-1, wf.getnchannels())
        sr = wf.getframerate()
    return resample(to_mono_float32(frames), sr, sample_rate)


def concat_with_gaps(parts: list[tuple[np.ndarray, int]], gap_s: float = 0.4) -> tuple[np.ndarray, int]:
    """Concatenate clips (possibly at different rates) into one track at the first clip's rate."""
    parts = [(a, sr) for a, sr in parts if a.size]
    if not parts:
        return np.zeros(0, dtype=np.float32), 22050
    out_sr = parts[0][1]
    gap = np.zeros(int(gap_s * out_sr), dtype=np.float32)
    chunks: list[np.ndarray] = []
    for audio, sr in parts:
        chunks.append(resample(audio, sr, out_sr))
        chunks.append(gap)
    return np.concatenate(chunks[:-1]), out_sr


# --------------------------------------------------------------------------- devices


def _sd():
    try:
        import sounddevice as sd
    except OSError as e:  # PortAudio shared library missing
        raise RuntimeError(f"sounddevice could not load PortAudio ({e}); install it with: sudo pacman -S portaudio") from e
    return sd


def list_devices() -> str:
    sd = _sd()
    lines = [str(sd.query_devices())]
    try:
        default_in, default_out = sd.default.device
        lines.append(f"\ndefault input: {default_in}   default output: {default_out}")
    except Exception:  # noqa: BLE001 - purely informational
        pass
    return "\n".join(lines)


def find_device(spec: str, kind: str) -> int | None:
    """Resolve a device index or name substring to a PortAudio index (``None`` = default)."""
    spec = str(spec).strip()
    if not spec:
        return None
    if spec.lstrip("-").isdigit():
        return int(spec)
    sd = _sd()
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = [
        i for i, d in enumerate(sd.query_devices()) if spec.lower() in d["name"].lower() and d[channel_key] > 0
    ]
    if not matches:
        raise ValueError(f"no {kind} device matches {spec!r}; run `./myvoice.py devices` to list them")
    if len(matches) > 1:
        log.warning("several %s devices match %r, using #%d", kind, spec, matches[0])
    return matches[0]


# --------------------------------------------------------------------------- streaming


class MicStream:
    """Captures mono float32 blocks from the microphone into a queue at ``sample_rate``.

    If the device refuses ``sample_rate`` it is opened at its native rate and every block is
    resampled before being queued.
    """

    def __init__(self, sample_rate: int = 16000, block_ms: int = 32, device: str = "") -> None:
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.device = find_device(device, "input")
        self.blocks: queue.Queue[np.ndarray] = queue.Queue(maxsize=2000)
        self._stream = None
        self._native_rate = sample_rate
        self.overflows = 0

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ARG002 - PortAudio signature
        if status and status.input_overflow:
            self.overflows += 1
        block = indata[:, 0].copy()
        if self._native_rate != self.sample_rate:
            block = resample(block, self._native_rate, self.sample_rate)
        try:
            self.blocks.put_nowait(block)
        except queue.Full:  # consumer stalled; drop audio rather than block the audio thread
            self.overflows += 1

    def start(self) -> None:
        sd = _sd()
        rates = [self.sample_rate]
        native = int(sd.query_devices(self.device, "input")["default_samplerate"])
        if native != self.sample_rate:
            rates.append(native)
        last_err: Exception | None = None
        for rate in rates:
            try:
                self._stream = sd.InputStream(
                    samplerate=rate,
                    channels=1,
                    dtype="float32",
                    blocksize=int(rate * self.block_ms / 1000),
                    device=self.device,
                    callback=self._callback,
                )
                self._native_rate = rate
                self._stream.start()
                if rate != self.sample_rate:
                    log.info("microphone opened at %d Hz (resampling to %d Hz)", rate, self.sample_rate)
                return
            except sd.PortAudioError as e:
                last_err = e
        raise RuntimeError(f"could not open the microphone: {last_err}")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class FileMic:
    """Feeds sound files into the pipeline as if they came from the microphone.

    Selected with ``audio.input_device = "file:a.wav,b.ogg"``: each ``start()`` plays the next
    file (cycling), in real time, followed by silence. Used for testing and demos.
    """

    def __init__(self, paths: list[Path], sample_rate: int = 16000, block_ms: int = 32, realtime: bool = True) -> None:
        if not paths:
            raise ValueError("file: microphone needs at least one file")
        self.paths = paths
        self.sample_rate = sample_rate
        self.block = int(sample_rate * block_ms / 1000)
        self.realtime = realtime
        self.blocks: queue.Queue[np.ndarray] = queue.Queue()
        self.overflows = 0
        self._next = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self, audio: np.ndarray) -> None:
        silence = np.zeros(self.block, dtype=np.float32)
        pos = 0
        while not self._stop.is_set():
            chunk = audio[pos : pos + self.block] if pos < audio.size else silence
            if chunk.size < self.block:
                chunk = np.concatenate([chunk, silence[: self.block - chunk.size]])
            pos += self.block
            self.blocks.put(chunk)
            if self.realtime:
                time.sleep(self.block / self.sample_rate)
            elif pos > audio.size + 5 * self.sample_rate:  # don't flood the queue with silence
                time.sleep(0.01)

    def start(self) -> None:
        audio = read_audio(self.paths[self._next % len(self.paths)], self.sample_rate)
        self._next += 1
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, args=(audio,), name="file-mic", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        while not self.blocks.empty():
            self.blocks.get_nowait()


def open_mic(sample_rate: int, block_ms: int, device: str) -> MicStream | FileMic:
    """A microphone stream, or a :class:`FileMic` for ``device = "file:a.wav,b.wav"``."""
    if device.startswith("file:"):
        paths = [Path(p).expanduser() for p in device[len("file:") :].split(",") if p.strip()]
        return FileMic(paths, sample_rate, block_ms)
    return MicStream(sample_rate, block_ms, device)


class Player:
    """Playback on the chosen output device: blocking (``play``) or in the background (``start``)."""

    def __init__(self, device: str = "") -> None:
        self.device = find_device(device, "output")

    def start(self, audio: np.ndarray, sample_rate: int) -> None:
        if audio.size == 0:
            return
        _sd().play(audio.astype(np.float32, copy=False), samplerate=sample_rate, device=self.device)

    def is_playing(self) -> bool:
        try:
            return bool(_sd().get_stream().active)
        except RuntimeError:  # nothing played yet
            return False

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        if audio.size == 0:
            return
        self.start(audio, sample_rate)
        _sd().wait()

    def stop(self) -> None:
        try:
            _sd().stop()
        except Exception:  # noqa: BLE001 - best effort during shutdown
            pass
