import json
import time

import numpy as np

from myvoice.asr import Transcript
from myvoice.audio import write_wav
from myvoice.config import load_config
from myvoice.pipeline import Engine, MicGate, SessionLog, format_turn, translate_file
from myvoice.vad import SAMPLE_RATE


class FakeAsr:
    def __init__(self, lang="it", text="Ciao, come stai?"):
        self.lang, self.text, self.calls = lang, text, []

    def transcribe(self, audio, allowed, forced_language=None):
        self.calls.append((audio.size, tuple(allowed), forced_language))
        lang = forced_language or self.lang
        return Transcript(self.text, lang, 0.97, -0.2, 0.01)


class FakeMt:
    def __init__(self):
        from collections import deque

        self.calls = []
        self.history = deque()

    def translate(self, text, src, tgt):
        self.calls.append((text, src, tgt))
        return f"<{src}->{tgt}> {text}"


class FakeTts:
    def synthesize(self, text, lang):
        return np.full(2205, 0.1, dtype=np.float32), 22050


def engine(**overrides):
    cfg = load_config(overrides=[f"{k}={v}" for k, v in overrides.items()])
    return Engine(cfg, FakeAsr(), FakeMt(), FakeTts()), cfg


def test_auto_mode_translates_into_the_other_language():
    eng, _ = engine()
    r = eng.process_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert (r.src_lang, r.tgt_lang) == ("it", "en")
    assert r.tgt_text == "<it->en> Ciao, come stai?"
    assert r.audio.size == 2205 and r.sample_rate == 22050
    assert set(r.timings) == {"asr", "mt", "tts"}
    assert eng.asr.calls[-1][2] is None  # language detection left to ASR

    eng.asr.lang = "en"
    r = eng.process_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert (r.src_lang, r.tgt_lang) == ("en", "it")


def test_fixed_mode_forces_asr_language():
    eng, _ = engine(**{"general.mode": '"en-it"'})
    r = eng.process_audio(np.zeros(SAMPLE_RATE, dtype=np.float32))
    assert eng.asr.calls[-1][2] == "en"
    assert (r.src_lang, r.tgt_lang) == ("en", "it")


def test_no_speech_returns_none():
    eng, _ = engine()
    eng.asr.transcribe = lambda *a, **k: None
    assert eng.process_audio(np.zeros(100, dtype=np.float32)) is None


def test_text_mode_guesses_language_and_can_skip_tts():
    eng, _ = engine()
    r = eng.process_text("Where is the station?", speak=False)
    assert (r.src_lang, r.tgt_lang) == ("en", "it")
    assert r.audio.size == 0
    assert "<en->it>" in format_turn(r)


def test_session_log_writes_jsonl(tmp_path):
    eng, cfg = engine(**{"general.session_log_dir": f'"{tmp_path}"'})
    log = SessionLog(cfg)
    log.write(eng.process_text("Ciao!", speak=False))
    rec = json.loads(log.path.read_text().strip())
    assert rec["src_text"] == "Ciao!" and "audio" not in rec


def test_mic_gate():
    g = MicGate()
    assert g.is_open()
    g.close()
    assert not g.is_open()
    g.open_after(0.05)
    assert not g.is_open()
    time.sleep(0.08)
    assert g.is_open()


def test_translate_file_end_to_end_with_fakes(tmp_path):
    eng, cfg = engine(**{"vad.backend": '"energy"', "vad.min_silence_ms": 300})
    t = np.arange(int(0.8 * SAMPLE_RATE)) / SAMPLE_RATE
    speech = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    gap = np.zeros(int(0.8 * SAMPLE_RATE), dtype=np.float32)
    src = write_wav(tmp_path / "in.wav", np.concatenate([gap, speech, gap, speech, gap]), SAMPLE_RATE)
    out = tmp_path / "out.wav"
    results = translate_file(eng, cfg, src, out)
    assert len(results) == 2
    assert out.exists() and out.with_suffix(".jsonl").exists()
    assert len(out.with_suffix(".jsonl").read_text().strip().splitlines()) == 2


def test_live_session_threads_with_fake_audio():
    import threading

    from myvoice.pipeline import LiveSession

    eng, cfg = engine(**{"vad.backend": '"energy"', "vad.min_silence_ms": 300,
                         "general.session_log_dir": '""', "audio.playback_tail_ms": 0})
    printed, played = [], []
    s = LiveSession(eng, cfg, printer=printed.append)
    s.mic.start = s.mic.stop = s.player.stop = lambda: None

    def fake_play(audio, sr):
        played.append((audio.size, sr, s.gate.is_open()))
        time.sleep(0.05)

    s.player.play = fake_play
    runner = threading.Thread(target=s.run, daemon=True)
    runner.start()
    t = np.arange(int(0.8 * SAMPLE_RATE)) / SAMPLE_RATE
    speech = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    quiet = np.zeros(int(0.8 * SAMPLE_RATE), dtype=np.float32)
    stream = np.concatenate([quiet, speech, quiet])
    for i in range(0, stream.size, 512):
        s.mic.blocks.put(stream[i : i + 512])
    deadline = time.monotonic() + 5
    while not played and time.monotonic() < deadline:
        time.sleep(0.02)
    s.stop.set()
    runner.join(timeout=3)
    assert not runner.is_alive()
    assert played and played[0][:2] == (2205, 22050)
    assert played[0][2] is False  # half-duplex: mic gate closed while playing
    assert any("<it->en> Ciao, come stai?" in p for p in printed)
