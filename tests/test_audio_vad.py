import numpy as np

from myvoice.audio import concat_with_gaps, float_to_int16, read_audio, resample, write_wav
from myvoice.config import VadConfig
from myvoice.vad import SAMPLE_RATE, EnergyVad, Segmenter

SR = SAMPLE_RATE


def tone(seconds: float, amp: float = 0.3, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), dtype=np.float32)


def test_resample_lengths_and_dtype():
    x = tone(1.0)
    y = resample(x, SR, 22050)
    assert y.dtype == np.float32
    assert abs(y.size - 22050) <= 1
    assert resample(x, SR, SR) is not None


def test_int16_conversion_clips():
    out = float_to_int16(np.array([-2.0, 0.0, 2.0], dtype=np.float32))
    assert out.tolist() == [-32767, 0, 32767]


def test_wav_roundtrip(tmp_path):
    x = tone(0.5)
    p = write_wav(tmp_path / "a.wav", x, SR)
    y = read_audio(p, SR)
    assert abs(y.size - x.size) <= 1
    assert np.max(np.abs(y[: x.size] - x[: y.size])) < 1e-3


def test_concat_with_gaps_resamples_to_first_rate():
    a = np.ones(1000, dtype=np.float32)
    b = np.ones(2000, dtype=np.float32)
    out, sr = concat_with_gaps([(a, 1000), (b, 2000)], gap_s=0.5)
    assert sr == 1000
    assert out.size == 1000 + 500 + 1000


def make_segmenter(**kw) -> Segmenter:
    cfg = VadConfig(backend="energy", min_silence_ms=400, pre_roll_ms=200, min_speech_ms=200, **kw)
    return Segmenter(EnergyVad(cfg.energy_threshold_db), cfg)


def feed_in_blocks(seg: Segmenter, audio: np.ndarray, block: int = 700):
    out = []
    for i in range(0, audio.size, block):  # odd block size: exercises frame re-chunking
        out.extend(seg.feed(audio[i : i + block]))
    return out


def test_segmenter_finds_two_utterances():
    audio = np.concatenate([silence(0.5), tone(1.0), silence(1.0), tone(0.8), silence(1.0)])
    seg = make_segmenter()
    utts = feed_in_blocks(seg, audio)
    assert len(utts) == 2
    first, second = utts
    assert 0.25 <= first.start_s <= 0.5  # includes pre-roll
    assert 1.0 <= first.duration_s <= 1.0 + 0.2 + 0.25 + 0.05  # speech + pre-roll + kept tail
    assert 2.2 <= second.start_s <= 2.5
    assert not seg.in_speech


def test_segmenter_ignores_short_blips_and_flushes_tail():
    audio = np.concatenate([silence(0.3), tone(0.05), silence(0.8), tone(0.6)])
    seg = make_segmenter()
    utts = feed_in_blocks(seg, audio)
    assert utts == []  # blip too short, second utterance not finished yet
    assert seg.in_speech
    last = seg.flush()
    assert last is not None and last.duration_s >= 0.6


def test_segmenter_forces_cut_on_long_speech():
    seg = make_segmenter(max_utterance_s=1.0)
    utts = feed_in_blocks(seg, tone(2.5))
    assert len(utts) == 2
    assert all(u.forced_cut for u in utts)


def test_segmenter_reset_drops_partial_speech():
    seg = make_segmenter()
    feed_in_blocks(seg, tone(0.5))
    assert seg.in_speech
    seg.reset()
    assert not seg.in_speech
    assert seg.flush() is None
