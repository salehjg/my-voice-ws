from pathlib import Path

import pytest

from myvoice.config import DEFAULT_CONFIG_PATH, load_config, parse_override
from myvoice.langs import guess_text_language, other_language, parse_mode


def test_default_config_file_loads_and_matches_dataclass_defaults():
    cfg = load_config()
    assert cfg.root == DEFAULT_CONFIG_PATH.parent
    assert cfg.general.languages == ["it", "en"]
    assert cfg.translation.backend == "llama_cpp"
    assert set(cfg.tts.voices) >= {"it", "en"}
    assert cfg.path(cfg.asr.model).is_absolute()


def test_overrides_are_parsed_as_toml_values():
    cfg = load_config(overrides=["asr.beam_size=1", "asr.compute_type=int8_float16", "audio.half_duplex=false",
                                 'general.mode="it-en"', "vad.threshold=0.6", "translation.n_ctx=4096"])
    assert cfg.asr.beam_size == 1
    assert cfg.asr.compute_type == "int8_float16"
    assert cfg.audio.half_duplex is False
    assert cfg.general.mode == "it-en"
    assert cfg.vad.threshold == 0.6
    assert parse_override("tts.voices.it=x.onnx") == {"tts": {"voices": {"it": "x.onnx"}}}


def test_int_is_accepted_for_float_fields():
    cfg = load_config(overrides=["tts.length_scale=1"])
    assert isinstance(cfg.tts.length_scale, float)


@pytest.mark.parametrize(
    "bad",
    ["asr.nope=1", "nosection.key=1", "asr.beam_size=fast", "audio.half_duplex=1", 'general.mode="it-fr"',
     'translation.backend="vllm"'],
)
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ValueError):
        load_config(overrides=[bad])


def test_relative_paths_resolve_against_config_dir(tmp_path: Path):
    p = tmp_path / "c.toml"
    p.write_text('[asr]\nmodel = "m/whisper"\n')
    cfg = load_config(p)
    assert cfg.path(cfg.asr.model) == tmp_path / "m" / "whisper"


def test_model_ref_keeps_hf_repo_ids(tmp_path: Path):
    cfg = load_config(overrides=['translation.model="Qwen/Qwen2.5-7B-Instruct"'])
    assert cfg.model_ref(cfg.translation.model) == "Qwen/Qwen2.5-7B-Instruct"
    assert cfg.model_ref("models/llm/x.gguf").endswith("models/llm/x.gguf")


def test_modes():
    assert parse_mode("auto", ["it", "en"]) is None
    for m in ("it-en", "it->en", "it2en", "IT-EN"):
        assert parse_mode(m, ["it", "en"]) == ("it", "en")
    assert parse_mode("en-it", ["it", "en"]) == ("en", "it")
    assert other_language("it", ["it", "en"]) == "en"
    with pytest.raises(ValueError):
        parse_mode("it-fr", ["it", "en"])


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Ciao, come stai? Vorrei un caffè per favore.", "it"),
        ("Dov'è la stazione dei treni?", "it"),
        ("Where is the train station, please?", "en"),
        ("I would like a coffee and a croissant.", "en"),
        ("Perché non andiamo al mare domani?", "it"),
    ],
)
def test_guess_text_language(text, expected):
    assert guess_text_language(text, ["it", "en"]) == expected
