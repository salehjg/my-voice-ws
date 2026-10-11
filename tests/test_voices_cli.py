import json

import numpy as np
import pytest
from typer.testing import CliRunner

from myvoice.audio import FileMic, write_wav
from myvoice.cli import app
from myvoice.config import LOCAL_CONFIG_NAME, load_config, save_local
from myvoice.tts import list_voices, resolve_piper, voice_language


def piper_voice(root, rel, lang, code):
    onnx = root / rel
    onnx.parent.mkdir(parents=True, exist_ok=True)
    onnx.write_bytes(b"onnx")
    (root / f"{rel}.json").write_text(json.dumps({"language": {"family": lang, "code": code}}))


@pytest.fixture
def cfg_dir(tmp_path):
    """A config dir with two Piper voices and a Kokoro voice pack."""
    piper_voice(tmp_path / "models/piper", "it/it_IT/paola/medium/it_IT-paola-medium.onnx", "it", "it_IT")
    piper_voice(tmp_path / "models/piper", "en/en_GB/alan/medium/en_GB-alan-medium.onnx", "en", "en_GB")
    (tmp_path / "models/kokoro").mkdir(parents=True)
    (tmp_path / "models/kokoro/kokoro-v1.0.onnx").write_bytes(b"x")
    np.savez(tmp_path / "models/kokoro/voices.npz", im_nicola=np.zeros(1), af_heart=np.zeros(1), zf_xiaobei=np.zeros(1))
    (tmp_path / "config.toml").write_text(
        '[tts]\nkokoro_voices = "models/kokoro/voices.npz"\n'
        '[tts.voices]\nit = "it_IT-paola-medium"\nen = "en_GB-alan-medium"\n'
    )
    return tmp_path


def test_list_voices(cfg_dir):
    cfg = load_config(cfg_dir / "config.toml")
    ids = {v.id: v for v in list_voices(cfg)}
    assert set(ids) == {"it_IT-paola-medium", "en_GB-alan-medium", "kokoro:im_nicola", "kokoro:af_heart"}
    assert ids["en_GB-alan-medium"].detail == "male · UK · medium"
    assert ids["kokoro:im_nicola"].lang == "it" and ids["kokoro:im_nicola"].name == "Nicola"
    assert [v.id for v in list_voices(cfg, "it")] == ["it_IT-paola-medium", "kokoro:im_nicola"]
    assert voice_language(cfg, "kokoro:af_heart") == "en"
    assert voice_language(cfg, "en_GB-alan-medium") == "en"
    assert voice_language(cfg, "nope") is None
    with pytest.raises(FileNotFoundError):
        resolve_piper(cfg, "it_IT-nobody-low")


def test_save_local_roundtrip(cfg_dir):
    cfg = load_config(cfg_dir / "config.toml")
    save_local(cfg, {"tts": {"voices": {"it": "kokoro:im_nicola"}}, "talk": {"speak_to_me": False}})
    assert cfg.tts.voices["it"] == "kokoro:im_nicola"
    save_local(cfg, {"tts": {"voices": {"en": "kokoro:af_heart"}}})
    again = load_config(cfg_dir / "config.toml", local=True)
    assert again.tts.voices == {"it": "kokoro:im_nicola", "en": "kokoro:af_heart"}
    assert again.talk.speak_to_me is False
    assert load_config(cfg_dir / "config.toml").tts.voices["it"] == "it_IT-paola-medium"  # local=False ignores it


def test_broken_local_config_names_the_file(cfg_dir):
    (cfg_dir / LOCAL_CONFIG_NAME).write_text("[talk]\nnope = 1\n")
    with pytest.raises(ValueError, match="config.local.toml"):
        load_config(cfg_dir / "config.toml", local=True)


def test_talk_languages_validated():
    with pytest.raises(ValueError, match="their_language"):
        load_config(overrides=['talk.their_language="en"'])


def test_file_mic_cycles_files(tmp_path):
    a = write_wav(tmp_path / "a.wav", np.full(1600, 0.5, np.float32), 16000)
    b = write_wav(tmp_path / "b.wav", np.full(1600, -0.5, np.float32), 16000)
    mic = FileMic([a, b], realtime=False)
    for expected in (0.5, -0.5, 0.5):
        mic.start()
        first = mic.blocks.get(timeout=2)
        mic.stop()
        assert first[0] == pytest.approx(expected, abs=1e-3)


runner = CliRunner()


def test_cli_help_lists_commands():
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in ("talk", "live", "file", "text", "voices", "check", "selftest"):
        assert cmd in r.output


def test_cli_config_error_exit_code():
    r = runner.invoke(app, ["check", "--set", "asr.nope=1"])
    assert r.exit_code == 2
    assert "config error" in r.output


def test_cli_voices_list_and_use(cfg_dir):
    cfg_path = str(cfg_dir / "config.toml")
    r = runner.invoke(app, ["voices", "-c", cfg_path])
    assert r.exit_code == 0, r.output
    assert "kokoro:im_nicola" in r.output and "★" in r.output
    r = runner.invoke(app, ["voices", "-c", cfg_path, "--use", "kokoro:im_nicola"])
    assert r.exit_code == 0, r.output
    assert load_config(cfg_path, local=True).tts.voices["it"] == "kokoro:im_nicola"
    r = runner.invoke(app, ["voices", "-c", cfg_path, "--use", "xx_nobody"])
    assert r.exit_code == 2
