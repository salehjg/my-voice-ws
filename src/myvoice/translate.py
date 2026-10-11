"""LLM translation: prompts, chat-template rendering, and two backends.

* ``llama_cpp``    – GGUF models via llama-cpp-python (CUDA). Lowest VRAM / latency. Default.
* ``transformers`` – Hugging Face checkpoints via PyTorch (optionally 4-bit bitsandbytes).

Prompt styles:

* ``translategemma`` – Google's TranslateGemma prompt (technical report, Fig. 3). We render the
  model's own chat template with the structured message it expects and fall back to a
  hand-built Gemma prompt with the same wording if the template is missing or fails.
* ``chat`` – a system prompt asking a general instruction model to act as an interpreter,
  optionally with the last few exchanges as context.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Config, TranslationConfig
from .langs import Language, get_language

log = logging.getLogger(__name__)

TRANSLATEGEMMA_INSTRUCTION = (
    "You are a professional {src_name} ({src_code}) to {tgt_name} ({tgt_code}) translator. "
    "Your goal is to accurately convey the meaning and nuances of the original {src_name} text "
    "while adhering to {tgt_name} grammar, vocabulary, and cultural sensitivities. "
    "Produce only the {tgt_name} translation, without any additional explanations or commentary. "
    "Please translate the following {src_name} text into {tgt_name}:\n\n\n{text}"
)

CHAT_SYSTEM_PROMPT = (
    "You are a professional simultaneous interpreter. Translate the user's {src_name} speech "
    "into natural, idiomatic spoken {tgt_name}. Preserve meaning, tone and register (formal "
    "Lei / informal tu), and keep names, numbers and units exact. The text comes from speech "
    "recognition, so it may lack punctuation or contain small recognition errors: translate "
    "what the speaker meant. Reply with the {tgt_name} translation only: no quotes, notes, "
    "alternatives or explanations."
)

STOP_STRINGS = ["<end_of_turn>", "<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<eos>"]


@dataclass
class Exchange:
    src_lang: str
    src_text: str
    tgt_lang: str
    tgt_text: str


# --------------------------------------------------------------------------- prompts


def translategemma_instruction(text: str, src: Language, tgt: Language) -> str:
    return TRANSLATEGEMMA_INSTRUCTION.format(
        src_name=src.name, src_code=src.region_code, tgt_name=tgt.name, tgt_code=tgt.region_code, text=text
    )


def translategemma_messages(text: str, src_code: str, tgt_code: str) -> list[dict[str, Any]]:
    """The structured single user turn TranslateGemma's chat template expects."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "source_lang_code": src_code, "target_lang_code": tgt_code, "text": text}
            ],
        }
    ]


def translategemma_manual_prompt(text: str, src: Language, tgt: Language) -> str:
    """Gemma-3 turn format with the official instruction (BOS is added by the tokenizer)."""
    return f"<start_of_turn>user\n{translategemma_instruction(text, src, tgt)}<end_of_turn>\n<start_of_turn>model\n"


def chat_messages(text: str, src: Language, tgt: Language, history: list[Exchange]) -> list[dict[str, str]]:
    system = CHAT_SYSTEM_PROMPT.format(src_name=src.name, tgt_name=tgt.name)
    if history:
        lines = []
        for ex in history:
            lines.append(f"- {get_language(ex.src_lang).name}: {ex.src_text}")
            lines.append(f"  {get_language(ex.tgt_lang).name}: {ex.tgt_text}")
        system += (
            "\n\nRecent conversation, for context only (do not translate it again):\n" + "\n".join(lines)
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


def render_chat_template(template: str, messages: list[dict[str, Any]], add_generation_prompt: bool = True, **extra: Any) -> str:
    """Render a Hugging Face style Jinja chat template (as stored in GGUF metadata).

    Mirrors the environment ``transformers`` uses so model-provided templates work unchanged.
    ``bos_token`` is rendered empty because the tokenizer adds BOS itself.
    """
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message: str) -> None:
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x: Any, ensure_ascii: bool = False, indent: int | None = None, separators=None, sort_keys: bool = False) -> str:
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
    context = {"bos_token": "", "eos_token": "", "add_generation_prompt": add_generation_prompt, **extra}
    return env.from_string(template).render(messages=messages, **context)


# --------------------------------------------------------------------------- output cleanup

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_LABEL_RE = re.compile(
    r"^\s*(?:translation|traduzione|english|italian|inglese|italiano|output|answer|risposta)"
    r"\s*(?:\([^)]*\))?\s*:\s*",
    re.I,
)
_NOTE_RE = re.compile(r"^\s*[\(\[]?\s*(?:note|nota|n\.b\.|explanation|spiegazione)\b", re.I)
_QUOTE_PAIRS = (('"', '"'), ("“", "”"), ("«", "»"), ("„", "“"), ("'", "'"))


def clean_output(raw: str, source: str = "") -> str:
    """Strip reasoning blocks, labels, notes and wrapping quotes from a model reply."""
    text = _THINK_RE.sub("", raw)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    if "<think>" in text:  # unterminated reasoning (ran out of tokens): nothing usable after it
        text = text.split("<think>")[0]
    for stop in STOP_STRINGS:
        text = text.split(stop)[0]
    lines: list[str] = []
    for line in text.strip().splitlines():
        if lines and _NOTE_RE.match(line):
            break
        if line.strip():
            lines.append(line.strip())
    text = " ".join(lines)
    text = _LABEL_RE.sub("", text, count=1).strip()
    src = source.strip()
    for left, right in _QUOTE_PAIRS:
        if len(text) >= 2 and text.startswith(left) and text.endswith(right) and not src.startswith(left):
            text = text[len(left) : -len(right)].strip()
            break
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- backends


class Translator(ABC):
    def __init__(self, cfg: TranslationConfig) -> None:
        self.cfg = cfg
        self.history: deque[Exchange] = deque(maxlen=max(0, cfg.context_turns))
        self._logged_prompt_path = False

    # -- backend hooks
    @abstractmethod
    def _render(self, messages: list[dict[str, Any]]) -> str:
        """Render messages with the model's chat template (raise if unavailable)."""

    @abstractmethod
    def _manual_translategemma(self, text: str, src: Language, tgt: Language) -> str: ...

    @abstractmethod
    def _complete(self, prompt: str, max_tokens: int) -> str: ...

    @abstractmethod
    def _chat_fallback(self, messages: list[dict[str, Any]], max_tokens: int) -> str: ...

    # -- shared logic
    def _note_path(self, what: str) -> None:
        if not self._logged_prompt_path:
            log.info("translation prompt: %s", what)
            self._logged_prompt_path = True

    def build_prompt(self, text: str, src: Language, tgt: Language) -> str | None:
        """Full prompt string, or ``None`` when the backend's own chat API must be used."""
        if self.cfg.prompt_style == "translategemma":
            for s, t in ((src.region_code, tgt.region_code), (src.code, tgt.code)):
                try:
                    prompt = self._render(translategemma_messages(text, s, t))
                    self._note_path(f"model chat template (TranslateGemma, codes {s}->{t})")
                    return prompt
                except Exception as e:  # noqa: BLE001 - any template problem -> next option
                    log.debug("TranslateGemma template with codes %s->%s failed: %s", s, t, e)
            self._note_path("built-in TranslateGemma prompt (model template unavailable)")
            return self._manual_translategemma(text, src, tgt)

        messages = chat_messages(text, src, tgt, list(self.history))
        try:
            prompt = self._render(messages)
            self._note_path("model chat template (interpreter system prompt)")
            return prompt
        except Exception as e:  # noqa: BLE001
            log.debug("chat template rendering failed: %s", e)
            self._note_path("backend chat API")
            return None

    def max_tokens_for(self, text: str) -> int:
        return min(self.cfg.max_new_tokens, 64 + 4 * len(text.split()))

    def translate(self, text: str, src_code: str, tgt_code: str) -> str:
        src, tgt = get_language(src_code), get_language(tgt_code)
        max_tokens = self.max_tokens_for(text)
        prompt = self.build_prompt(text, src, tgt)
        if prompt is not None:
            log.debug("prompt:\n%s", prompt)
            raw = self._complete(prompt, max_tokens)
        else:
            raw = self._chat_fallback(chat_messages(text, src, tgt, list(self.history)), max_tokens)
        log.debug("raw model output: %r", raw)
        out = clean_output(raw, text)
        if out and self.history.maxlen:
            self.history.append(Exchange(src_code, text, tgt_code, out))
        return out


@contextlib.contextmanager
def quiet_native_stderr():
    """Silence C-level stderr (llama.cpp/ggml load messages) unless debug logging is on."""
    if log.isEnabledFor(logging.DEBUG):
        yield
        return
    saved = os.dup(2)
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


class LlamaCppTranslator(Translator):
    def __init__(self, cfg: TranslationConfig, model_path: str | Path) -> None:
        super().__init__(cfg)
        with quiet_native_stderr():
            from llama_cpp import Llama

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"GGUF model not found: {path} (see to-download.md)")
        kwargs: dict[str, Any] = {
            "model_path": str(path),
            "n_gpu_layers": cfg.n_gpu_layers,
            "n_ctx": cfg.n_ctx,
            "verbose": False,
        }
        if cfg.flash_attn:
            kwargs["flash_attn"] = True
        with quiet_native_stderr():
            self.llm = Llama(**kwargs)
        self.template: str | None = (self.llm.metadata or {}).get("tokenizer.chat_template")
        self._bos = self.llm.token_bos()

    def _render(self, messages: list[dict[str, Any]]) -> str:
        if not self.template:
            raise LookupError("GGUF file has no tokenizer.chat_template")
        return render_chat_template(self.template, messages, enable_thinking=False)

    def _manual_translategemma(self, text: str, src: Language, tgt: Language) -> str:
        return translategemma_manual_prompt(text, src, tgt)

    def _complete(self, prompt: str, max_tokens: int) -> str:
        tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=True)
        # A template that hard-codes BOS plus the tokenizer's own BOS would give two.
        while len(tokens) > 1 and self._bos >= 0 and tokens[0] == tokens[1] == self._bos:
            tokens = tokens[1:]
        room = self.cfg.n_ctx - len(tokens)
        if room < 16:
            raise ValueError(f"prompt is {len(tokens)} tokens; increase translation.n_ctx (now {self.cfg.n_ctx})")
        out = self.llm.create_completion(
            tokens,
            max_tokens=min(max_tokens, room),
            temperature=self.cfg.temperature,
            top_p=1.0,
            repeat_penalty=1.0,
            stop=STOP_STRINGS,
        )
        return out["choices"][0]["text"]

    def _chat_fallback(self, messages: list[dict[str, Any]], max_tokens: int) -> str:
        out = self.llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=self.cfg.temperature, repeat_penalty=1.0
        )
        return out["choices"][0]["message"]["content"] or ""


class TransformersTranslator(Translator):
    def __init__(self, cfg: TranslationConfig, model_ref: str) -> None:
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_ref, local_files_only=True)
        dtype = getattr(torch, cfg.dtype)
        kwargs: dict[str, Any] = {"dtype": dtype, "device_map": "cuda:0", "local_files_only": True}
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_ref, **kwargs)
        except ValueError:
            # Multimodal checkpoints such as TranslateGemma (Gemma 3) register as image-text-to-text.
            from transformers import AutoModelForImageTextToText

            self.model = AutoModelForImageTextToText.from_pretrained(model_ref, **kwargs)
        self.model.eval()

    def _render(self, messages: list[dict[str, Any]]) -> str:
        if not getattr(self.tok, "chat_template", None):
            raise LookupError("tokenizer has no chat template")
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _manual_translategemma(self, text: str, src: Language, tgt: Language) -> str:
        return (self.tok.bos_token or "") + translategemma_manual_prompt(text, src, tgt)

    def _complete(self, prompt: str, max_tokens: int) -> str:
        torch = self._torch
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        gen: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": self.cfg.temperature > 0,
            "pad_token_id": self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id,
        }
        if self.cfg.temperature > 0:
            gen["temperature"] = self.cfg.temperature
        else:
            gen.update(top_p=None, top_k=None, temperature=None)
        with torch.inference_mode():
            out = self.model.generate(**enc, **gen)
        return self.tok.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)

    def _chat_fallback(self, messages: list[dict[str, Any]], max_tokens: int) -> str:
        raise RuntimeError("this tokenizer has no chat template; use prompt_style = 'translategemma' or another model")


def make_translator(cfg: Config) -> Translator:
    t = cfg.translation
    if t.backend == "llama_cpp":
        return LlamaCppTranslator(t, cfg.path(t.model))
    return TransformersTranslator(t, cfg.model_ref(t.model))
