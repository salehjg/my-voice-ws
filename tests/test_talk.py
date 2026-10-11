import io
import queue

import numpy as np
import pytest
from rich.console import Console

from myvoice.asr import Transcript
from myvoice.audio import write_wav
from myvoice.config import load_config
from myvoice.pipeline import Engine
from myvoice.talk import ScriptedKeys, TalkApp, decode_key, record_utterance
from myvoice.tts import VoiceInfo
from myvoice.vad import SAMPLE_RATE


def tone(seconds=0.8, amp=0.3):
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class FakeMic:
    """Each start() queues the next clip followed by 3 s of silence. The clips cycle; with
    ``cycle=False`` each plays once and later starts give silence only."""

    def __init__(self, clips, cycle=True):
        self.clips, self.n, self.cycle = clips, 0, cycle
        self.blocks = queue.Queue()
        self.starts = self.stops = 0

    def start(self):
        if self.cycle or self.n < len(self.clips):
            clip = self.clips[self.n % len(self.clips)]
        else:
            clip = np.zeros(0, np.float32)
        self.n += 1
        self.starts += 1
        audio = np.concatenate([np.zeros(4000, np.float32), clip, np.zeros(3 * SAMPLE_RATE, np.float32)])
        for i in range(0, audio.size, 512):
            self.blocks.put(audio[i : i + 512])

    def stop(self):
        self.stops += 1
        while not self.blocks.empty():
            self.blocks.get_nowait()


class FakeAsr:
    def __init__(self, texts=None):
        self.texts = texts or {"it": "Buongiorno, come sta?", "en": "Very well, thank you."}
        self.calls = []

    def transcribe(self, audio, allowed, forced_language=None):
        self.calls.append(forced_language)
        lang = forced_language or "it"
        return Transcript(self.texts[lang], lang, 0.99, -0.1, 0.01)


class DetectAsr(FakeAsr):
    """Detects the languages in ``detect``, one per call (a forced language is used as given)."""

    def __init__(self, detect, prob=0.99):
        super().__init__()
        self.detect, self.prob = list(detect), prob

    def transcribe(self, audio, allowed, forced_language=None):
        self.calls.append(forced_language)
        lang = forced_language or self.detect.pop(0)
        return Transcript(self.texts[lang], lang, 1.0 if forced_language else self.prob, -0.1, 0.01)


class FakeMt:
    def __init__(self):
        from collections import deque

        self.calls, self.history = [], deque()

    def translate(self, text, src, tgt):
        self.calls.append((text, src, tgt))
        return f"<{tgt}> {text}"


class FakeTts:
    def __init__(self):
        self.voices = {"it": "it_IT-paola-medium", "en": "en_US-lessac-high"}
        self.spoken = []

    def synthesize(self, text, lang, voice_id=None):
        self.spoken.append((lang, text))
        return np.full(2205, 0.1, dtype=np.float32), 22050

    def set_voice(self, lang, voice_id):
        self.voices[lang] = voice_id


class FakePlayer:
    def __init__(self):
        self.played = []

    def start(self, audio, sr):
        self.played.append((audio.size, sr))

    def is_playing(self):
        return False

    def play(self, audio, sr):
        self.start(audio, sr)

    def stop(self):
        pass


def make_app(keys, clips=None, **overrides):
    sets = {"vad.backend": '"energy"', "general.session_log_dir": '""', **overrides}
    cfg = load_config(overrides=[f"{k}={v}" for k, v in sets.items()])
    engine = Engine(cfg, FakeAsr(), FakeMt(), FakeTts())
    out = io.StringIO()
    mic = FakeMic(clips or [tone()])
    app = TalkApp(engine, cfg, ScriptedKeys(keys), console=Console(file=out, width=120),
                  player=FakePlayer(), mic_factory=lambda: mic, save_choices=False)
    return app, out, mic


@pytest.mark.parametrize("raw,key", [
    (b" ", "space"), (b"\r", "enter"), (b"\n", "enter"), (b"\x1b", "esc"), (b"\x1b[A", "up"),
    (b"Q", "q"), (b"e", "e"), (b"?", "?"), (b"\x7f", "backspace"), (b"\x04", "ctrl-d"), (b"\t", "tab"), (b"", None),
])
def test_decode_key(raw, key):
    assert decode_key(raw) == key


def test_scripted_keys_polls_do_not_consume():
    k = ScriptedKeys(["space", "q"])
    assert k.get(timeout=0) is None
    assert k.get() == "space"
    assert k.get(timeout=2) == "q"
    assert k.get() == "q"  # exhausted -> quit


def test_record_utterance_stops_on_silence():
    cfg = load_config(overrides=['vad.backend="energy"'])
    mic = FakeMic([tone(1.0)])
    levels = []
    rec = record_utterance(cfg, ScriptedKeys([]), 600, mic, lambda db, t, s: levels.append(db))
    assert rec is not None and rec.stopped_by == "silence"
    assert 0.9 < rec.seconds < 2.0
    assert mic.starts == mic.stops == 1
    assert max(levels) > -20


def test_record_utterance_times_out_without_speech():
    cfg = load_config(overrides=['vad.backend="energy"', "talk.no_speech_timeout_s=0.3"])
    mic = FakeMic([np.zeros(100, np.float32)])
    assert record_utterance(cfg, ScriptedKeys([]), 600, mic) is None


def test_full_cycle_listen_translate_answer_speak():
    # SPACE: listen to Italian -> English; SPACE: record English; ENTER: confirm -> spoken Italian
    app, out, mic = make_app(["space", "space", "enter", "q"], clips=[tone(0.8), tone(1.6)])
    app.run()
    assert mic.n == 2  # one mic object, started once per recording
    eng = app.engine
    assert eng.asr.calls == ["it", "en"]  # languages forced per step, no detection
    assert eng.mt.calls == [("Buongiorno, come sta?", "it", "en"), ("Very well, thank you.", "en", "it")]
    assert eng.tts.spoken == [("en", "<en> Buongiorno, come sta?"), ("it", "<it> Very well, thank you.")]
    assert len(app.player.played) == 2
    text = out.getvalue()
    assert "They said" in text and "You said" in text and "Your turn" in text


def test_english_audio_off_only_prints():
    app, _, _ = make_app(["space", "s", "q"], **{"talk.speak_to_me": "false"})
    app.run()
    assert app.engine.tts.spoken == [] and app.player.played == []
    assert app.engine.mt.calls == [("Buongiorno, come sta?", "it", "en")]


def test_confirm_edit_and_record_again():
    # E: answer directly; R: record again; E: edit the transcript; translated text is the edited one
    app, _, mic = make_app(["e", "r", "e", "Very well, and you?", "q"])
    app.run()
    assert mic.starts == 2
    assert app.engine.mt.calls == [("Very well, and you?", "en", "it")]


def test_confirm_cancel_returns_to_your_turn():
    app, out, _ = make_app(["e", "esc", "q"])
    app.run()
    assert app.engine.mt.calls == []
    assert "(cancelled)" in out.getvalue()


def test_no_confirm_translates_immediately():
    app, _, _ = make_app(["e", "q"], **{"talk.confirm_my_speech": "false"})
    app.run()
    assert app.engine.mt.calls == [("Very well, thank you.", "en", "it")]


def test_type_both_languages_and_replay():
    app, _, _ = make_app(["t", "See you tomorrow", "i", "A domani", "r", "q"])
    app.run()
    assert app.engine.mt.calls == [("See you tomorrow", "en", "it"), ("A domani", "it", "en")]
    assert len(app.player.played) == 3  # Italian speech, English speech, replay


def test_toggles_change_config():
    app, out, _ = make_app(["a", "c", "h", "q"])
    app.run()
    assert app.cfg.talk.speak_to_me is False and app.cfg.talk.confirm_my_speech is False
    assert "change voices" in out.getvalue()  # help table


def test_voices_menu_cycles_and_plays(monkeypatch):
    fake = [VoiceInfo("it_IT-paola-medium", "piper", "it", "Paola", "female"),
            VoiceInfo("kokoro:im_nicola", "kokoro", "it", "Nicola", "male"),
            VoiceInfo("en_US-lessac-high", "piper", "en", "Lessac", "female")]
    monkeypatch.setattr("myvoice.tts.list_voices", lambda cfg, lang=None, all_kokoro=False:
                        [v for v in fake if lang is None or v.lang == lang])
    app, _, _ = make_app(["v", "i", "esc", "q"])
    app.run()
    assert app.engine.tts.voices["it"] == "kokoro:im_nicola"
    assert app.cfg.tts.voices["it"] == "kokoro:im_nicola"
    assert app.engine.tts.spoken[-1][0] == "it"  # sample played in the new voice


def test_translate_file(tmp_path):
    path = write_wav(tmp_path / "note.wav", np.concatenate([np.zeros(8000, np.float32), tone(), np.zeros(16000, np.float32)]),
                     SAMPLE_RATE)
    app, out, _ = make_app(["f", f"'{path}'", "q"])
    app.run()
    assert app.engine.mt.calls == [("Buongiorno, come sta?", "it", "en")]
    assert "note.wav" in out.getvalue()


def test_translate_missing_file():
    app, out, _ = make_app(["f", "/nope/missing.ogg", "q"])
    app.run()
    assert "no such file" in out.getvalue()


class QuietMic(FakeMic):
    """Speech too quiet for the energy VAD; nothing ends the recording except a key."""

    def start(self):
        self.starts += 1
        for _ in range(60):
            self.blocks.put(np.full(512, 0.001, np.float32))


class KeyAfter(ScriptedKeys):
    """Returns ``key`` on the n-th poll (to stop a recording by hand)."""

    def __init__(self, n, key):
        super().__init__([])
        self.n, self.key = n, key

    def get(self, timeout=None):
        self.n -= 1
        return self.key if self.n == 0 else None


def test_manual_stop_keeps_audio_vad_did_not_hear():
    cfg = load_config(overrides=['vad.backend="energy"'])
    rec = record_utterance(cfg, KeyAfter(40, "space"), 600, QuietMic([]))
    assert rec is not None and rec.stopped_by == "key" and rec.audio.size > SAMPLE_RATE // 2


def test_escape_cancels_recording():
    cfg = load_config(overrides=['vad.backend="energy"'])
    assert record_utterance(cfg, KeyAfter(5, "esc"), 600, FakeMic([tone()])) is None


def test_multi_sentence_reply_is_spoken_sentence_by_sentence():
    app, _, _ = make_app(["t", "It costs six euros. The bus leaves at nine! Have a good trip.", "r", "q"])
    app.run()
    spoken = [t for lang, t in app.engine.tts.spoken if lang == "it"]
    assert spoken == ["<it> It costs six euros.", "The bus leaves at nine!", "Have a good trip."]
    assert len(app.player.played) == 4  # three sentences + one replay of the joined audio
    assert app.player.played[-1][0] > 3 * 2205  # replay = all sentences (+ gaps)


def test_errors_are_shown_and_the_app_keeps_going():
    app, out, _ = make_app(["x", "t", "Hello", "space", "q"])

    def broken(*a, **k):
        raise RuntimeError("voice exploded")

    app.engine.tts.synthesize = broken
    app.run()
    text = out.getvalue()
    assert "voice exploded" in text
    assert text.count("voice exploded") == 2  # typed answer + their next turn, both survived
    assert app.engine.mt.calls == [("Hello", "en", "it"), ("Buongiorno, come sta?", "it", "en")]


# --------------------------------------------------------------------------- auto mode


class AutoKeys(ScriptedKeys):
    """Auto mode only polls for keys: the next key comes once the microphone has nothing left."""

    def __init__(self, keys, mic):
        super().__init__(keys)
        self.mic = mic

    def get(self, timeout=None):
        if timeout is not None and timeout < 0.5 and not self.mic.blocks.empty():
            return None
        return self.keys.pop(0) if self.keys else "q"


def make_auto_app(keys, clips, detect=(), prob=0.99, **overrides):
    sets = {"vad.backend": '"energy"', "general.session_log_dir": '""', "audio.playback_tail_ms": 0, **overrides}
    cfg = load_config(overrides=[f"{k}={v}" for k, v in sets.items()])
    engine = Engine(cfg, DetectAsr(detect, prob), FakeMt(), FakeTts())
    out = io.StringIO()
    mic = FakeMic(clips, cycle=False)
    app = TalkApp(engine, cfg, AutoKeys(keys, mic), console=Console(file=out, width=140),
                  player=FakePlayer(), mic_factory=lambda: mic, save_choices=False)
    return app, out, mic


def gap(seconds=2.0):
    return np.zeros(int(seconds * SAMPLE_RATE), np.float32)


def test_auto_mode_translates_whatever_it_hears():
    # Italian -> spoken in English; mic paused while speaking; English -> spoken in Italian
    app, out, mic = make_auto_app(["q"], [tone(0.8), tone(1.6)], detect=["it", "en"])
    app.run("auto")
    eng = app.engine
    assert eng.asr.calls == [None, None]  # language detected, not forced
    assert eng.mt.calls == [("Buongiorno, come sta?", "it", "en"), ("Very well, thank you.", "en", "it")]
    assert eng.tts.spoken == [("en", "<en> Buongiorno, come sta?"), ("it", "<it> Very well, thank you.")]
    assert len(app.player.played) == 2
    assert mic.starts == 3 and mic.stops == 3  # restarted after each translation; stopped on quit
    text = out.getvalue()
    assert "auto mode" in text and "Italian → English" in text and "English → Italian" in text


def test_auto_mode_with_headphones_keeps_listening():
    app, _, mic = make_auto_app(["q"], [np.concatenate([tone(0.8), gap(), tone(1.6)])], detect=["it", "en"],
                                **{"audio.half_duplex": "false"})
    app.run("auto")
    assert len(app.engine.mt.calls) == 2 and len(app.player.played) == 2
    assert mic.starts == 1  # never paused


def test_auto_mode_my_audio_off_only_speaks_their_language():
    app, _, _ = make_auto_app(["q"], [np.concatenate([tone(0.8), gap(), tone(1.6)])], detect=["it", "en"],
                              **{"talk.speak_to_me": "false"})
    app.run("auto")
    assert len(app.engine.mt.calls) == 2
    assert app.engine.tts.spoken == [("it", "<it> Very well, thank you.")]


def test_auto_mode_wrong_language_redo_and_unsure_hint():
    app, out, _ = make_auto_app(["w", "q"], [tone(0.8)], detect=["it"], prob=0.6)
    app.run("auto")
    assert app.engine.asr.calls == [None, "en"]  # redone as the other language
    assert app.engine.mt.calls[-1] == ("Very well, thank you.", "en", "it")
    text = out.getvalue()
    assert "60% sure" in text and "Redone" in text


def test_auto_mode_type_replay_and_help():
    app, out, _ = make_auto_app(["t", "Ciao a tutti", "r", "h", "q"], [np.zeros(100, np.float32)])
    app.run("auto")
    assert app.engine.mt.calls == [("Ciao a tutti", "it", "en")]  # language guessed from the words
    assert len(app.player.played) == 2  # spoken + replay
    assert "wrong language" in out.getvalue()  # auto mode help


def test_pause_then_switch_modes_both_ways():
    # auto: SPACE pauses, TAB -> manual; manual: TAB -> auto; Q quits
    app, out, mic = make_auto_app(["space", "tab", "tab", "q"], [np.zeros(100, np.float32)])
    app.run("auto")
    text = out.getvalue()
    assert "Paused" in text
    assert text.index("auto mode ·") < text.index("manual mode ·") < text.rindex("auto mode ·")
    assert mic.starts == 2 and mic.stops == 2  # the mic is off in manual mode until you record
    assert app.engine.mt.calls == []
