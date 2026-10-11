"""Configuration: typed defaults, TOML loading, ``--set`` overrides and path resolution."""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"


@dataclass
class GeneralConfig:
    mode: str = "auto"
    languages: list[str] = field(default_factory=lambda: ["it", "en"])
    session_log_dir: str = "outputs"
    warmup: bool = True


@dataclass
class AudioConfig:
    input_device: str = ""
    output_device: str = ""
    sample_rate: int = 16000
    block_ms: int = 32
    half_duplex: bool = True
    playback_tail_ms: int = 300
    max_wait_for_silence_s: float = 8.0


@dataclass
class VadConfig:
    backend: str = "silero"
    threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 700
    pre_roll_ms: int = 300
    max_utterance_s: float = 25.0
    energy_threshold_db: float = -42.0


@dataclass
class AsrConfig:
    model: str = "models/whisper/faster-whisper-large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    beam_size: int = 5
    initial_prompt: str = ""
    filter_hallucinations: bool = True


@dataclass
class TranslationConfig:
    backend: str = "llama_cpp"
    model: str = "models/llm/translategemma-4b-it.Q6_K.gguf"
    prompt_style: str = "translategemma"
    n_gpu_layers: int = -1
    n_ctx: int = 2048
    flash_attn: bool = True
    max_new_tokens: int = 384
    temperature: float = 0.0
    context_turns: int = 2
    load_in_4bit: bool = True
    dtype: str = "bfloat16"


@dataclass
class TtsConfig:
    backend: str = "piper"
    use_cuda: bool = False
    length_scale: float = 1.0
    volume: float = 1.0
    sentence_silence_ms: int = 150
    voices: dict[str, str] = field(
        default_factory=lambda: {
            "it": "models/piper/it/it_IT/paola/medium/it_IT-paola-medium.onnx",
            "en": "models/piper/en/en_US/lessac/high/en_US-lessac-high.onnx",
        }
    )


@dataclass
class Config:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    root: Path = PROJECT_ROOT

    def path(self, value: str) -> Path:
        """Resolve a path from the config relative to the config file's directory."""
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.root / p)

    def model_ref(self, value: str) -> str:
        """Like :meth:`path`, but keeps Hugging Face repo ids (``org/name``) that are not local dirs.

        This lets the transformers backend load a model straight from the HF cache.
        """
        p = self.path(value)
        if p.exists():
            return str(p)
        parts = value.split("/")
        if len(parts) == 2 and not value.startswith((".", "/", "~")) and all(parts):
            return value
        return str(p)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["root"] = str(self.root)
        return d


_SECTIONS = {f.name: f.type for f in dataclasses.fields(Config) if f.name != "root"}


def _merge_section(section_obj: Any, values: dict[str, Any], section: str) -> None:
    known = {f.name for f in dataclasses.fields(section_obj)}
    for key, value in values.items():
        if key not in known:
            raise ValueError(f"unknown config key [{section}] {key!r} (known: {', '.join(sorted(known))})")
        current = getattr(section_obj, key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged = dict(current)
            merged.update(value)
            value = merged
        elif isinstance(current, bool) and not isinstance(value, bool):
            raise ValueError(f"[{section}] {key} must be true/false, got {value!r}")
        elif isinstance(current, float) and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        elif current is not None and not isinstance(value, type(current)):
            raise ValueError(f"[{section}] {key} must be {type(current).__name__}, got {value!r}")
        setattr(section_obj, key, value)


def _apply(cfg: Config, data: dict[str, Any]) -> None:
    for section, values in data.items():
        if section not in _SECTIONS:
            raise ValueError(f"unknown config section [{section}] (known: {', '.join(_SECTIONS)})")
        if not isinstance(values, dict):
            raise ValueError(f"config section [{section}] must be a table")
        _merge_section(getattr(cfg, section), values, section)


def parse_override(item: str) -> dict[str, Any]:
    """Turn ``"asr.beam_size=1"`` into ``{"asr": {"beam_size": 1}}``.

    The value is parsed as a TOML value when possible (numbers, booleans, arrays, quoted
    strings); otherwise it is taken as a bare string.
    """
    if "=" not in item:
        raise ValueError(f"override {item!r} must look like section.key=value")
    key, raw = item.split("=", 1)
    parts = key.strip().split(".")
    if len(parts) < 2:
        raise ValueError(f"override key {key!r} must look like section.key")
    try:
        value = tomllib.loads(f"v = {raw.strip()}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw.strip()
    out: dict[str, Any] = {}
    node = out
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


def load_config(path: str | Path | None = None, overrides: Iterable[str] = ()) -> Config:
    cfg = Config()
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if cfg_path.exists():
        with open(cfg_path, "rb") as f:
            _apply(cfg, tomllib.load(f))
        cfg.root = cfg_path.resolve().parent
    elif path:
        raise FileNotFoundError(f"config file not found: {cfg_path}")
    for item in overrides:
        _apply(cfg, parse_override(item))
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    from .langs import get_language, parse_mode

    langs = cfg.general.languages
    if len(langs) != 2 or langs[0] == langs[1]:
        raise ValueError(f"general.languages must name two different languages, got {langs}")
    for code in langs:
        get_language(code)
    parse_mode(cfg.general.mode, langs)
    if cfg.translation.backend not in ("llama_cpp", "transformers"):
        raise ValueError(f"translation.backend must be llama_cpp or transformers, got {cfg.translation.backend!r}")
    if cfg.translation.prompt_style not in ("translategemma", "chat"):
        raise ValueError(f"translation.prompt_style must be translategemma or chat, got {cfg.translation.prompt_style!r}")
    if cfg.vad.backend not in ("silero", "energy"):
        raise ValueError(f"vad.backend must be silero or energy, got {cfg.vad.backend!r}")
    if cfg.tts.backend != "piper":
        raise ValueError(f"tts.backend must be piper, got {cfg.tts.backend!r}")
    missing = [code for code in langs if code not in cfg.tts.voices]
    if missing:
        raise ValueError(f"tts.voices has no voice for: {', '.join(missing)}")
