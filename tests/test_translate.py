import pytest

from myvoice.asr import choose_language, is_hallucination
from myvoice.config import TranslationConfig
from myvoice.langs import get_language
from myvoice.translate import (
    Translator,
    chat_messages,
    clean_output,
    render_chat_template,
    translategemma_instruction,
    translategemma_manual_prompt,
)

IT, EN = get_language("it"), get_language("en")

# A cut-down Gemma-3 style template that, like TranslateGemma's, reads structured content.
GEMMA_LIKE_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<start_of_turn>{{ 'model' if m.role == 'assistant' else m.role }}\n"
    "{% if m.content is string %}{{ m.content }}{% else %}{% for c in m.content %}"
    "{% if c.source_lang_code not in ['it', 'en', 'it-IT', 'en-US'] %}{{ raise_exception('bad code') }}{% endif %}"
    "[{{ c.source_lang_code }}>{{ c.target_lang_code }}] {{ c.text }}{% endfor %}{% endif %}<end_of_turn>\n"
    "{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


def test_translategemma_instruction_matches_official_wording():
    s = translategemma_instruction("Ciao!", IT, EN)
    assert s.startswith("You are a professional Italian (it-IT) to English (en-US) translator.")
    assert "Produce only the English translation, without any additional explanations or commentary." in s
    assert s.endswith("Please translate the following Italian text into English:\n\n\nCiao!")
    p = translategemma_manual_prompt("Ciao!", IT, EN)
    assert p.startswith("<start_of_turn>user\nYou are") and p.endswith("Ciao!<end_of_turn>\n<start_of_turn>model\n")


def test_render_chat_template_structured_and_plain():
    msgs = [{"role": "user", "content": [{"type": "text", "source_lang_code": "it", "target_lang_code": "en", "text": "Ciao"}]}]
    out = render_chat_template(GEMMA_LIKE_TEMPLATE, msgs)
    assert out == "<start_of_turn>user\n[it>en] Ciao<end_of_turn>\n<start_of_turn>model\n"
    plain = render_chat_template(GEMMA_LIKE_TEMPLATE, [{"role": "user", "content": "hi"}], add_generation_prompt=False)
    assert plain == "<start_of_turn>user\nhi<end_of_turn>\n"


def test_render_chat_template_raise_exception():
    from jinja2.exceptions import TemplateError

    msgs = [{"role": "user", "content": [{"type": "text", "source_lang_code": "xx", "target_lang_code": "en", "text": "?"}]}]
    with pytest.raises(TemplateError):
        render_chat_template(GEMMA_LIKE_TEMPLATE, msgs)


@pytest.mark.parametrize(
    "raw,source,expected",
    [
        ("Good morning, how are you?", "Buongiorno", "Good morning, how are you?"),
        ("  \"Good morning.\"  ", "Buongiorno.", "Good morning."),
        ("“Buongiorno.”", "Good morning.", "Buongiorno."),
        ('"Hi," she said.', '"Ciao," disse.', '"Hi," she said.'),
        ("<think>\nthe user wants...\n</think>\n\nHello!", "Ciao!", "Hello!"),
        ("Translation: Hello!", "Ciao!", "Hello!"),
        ("Hello there!\n\nNote: 'Ciao' can mean hi or bye.", "Ciao!", "Hello there!"),
        ("Ciao!<end_of_turn>junk", "Hi!", "Ciao!"),
        ("Line one.\nLine two.", "Uno. Due.", "Line one. Line two."),
        ("<think>never finished", "x", ""),
    ],
)
def test_clean_output(raw, source, expected):
    assert clean_output(raw, source) == expected


def test_chat_messages_include_context():
    from myvoice.translate import Exchange

    msgs = chat_messages("Sì, certo.", IT, EN, [Exchange("en", "Can you help me?", "it", "Mi può aiutare?")])
    assert msgs[0]["role"] == "system" and "Italian speech" in msgs[0]["content"]
    assert "English: Can you help me?" in msgs[0]["content"]
    assert msgs[1] == {"role": "user", "content": "Sì, certo."}


class FakeTranslator(Translator):
    def __init__(self, cfg, template_ok=True, reply="Hello!"):
        super().__init__(cfg)
        self.template_ok = template_ok
        self.reply = reply
        self.prompts = []

    def _render(self, messages):
        if not self.template_ok:
            raise LookupError("no template")
        return render_chat_template(GEMMA_LIKE_TEMPLATE, messages)

    def _manual_translategemma(self, text, src, tgt):
        return "MANUAL:" + translategemma_manual_prompt(text, src, tgt)

    def _complete(self, prompt, max_tokens):
        self.prompts.append(prompt)
        return self.reply

    def _chat_fallback(self, messages, max_tokens):
        self.prompts.append(("chat", messages))
        return self.reply


def test_translategemma_uses_template_then_falls_back():
    t = FakeTranslator(TranslationConfig(prompt_style="translategemma"))
    assert t.translate("Ciao!", "it", "en") == "Hello!"
    assert t.prompts[-1].startswith("<start_of_turn>user\n[it-IT>en-US] Ciao!")

    t2 = FakeTranslator(TranslationConfig(prompt_style="translategemma"), template_ok=False)
    t2.translate("Ciao!", "it", "en")
    assert t2.prompts[-1].startswith("MANUAL:<start_of_turn>user\nYou are a professional Italian")


def test_chat_style_history_and_fallback():
    cfg = TranslationConfig(prompt_style="chat", context_turns=1)
    t = FakeTranslator(cfg, template_ok=False, reply="«Ciao!»")
    assert t.translate("Hi!", "en", "it") == "Ciao!"
    assert t.prompts[-1][0] == "chat"
    t.translate("Bye!", "en", "it")
    system = t.prompts[-1][1][0]["content"]
    assert "English: Hi!" in system  # previous exchange given as context
    assert len(t.history) == 1  # context_turns=1


def test_max_tokens_scales_with_input():
    t = FakeTranslator(TranslationConfig(max_new_tokens=100))
    assert t.max_tokens_for("uno due tre") == 64 + 12
    assert t.max_tokens_for("parola " * 200) == 100


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Sottotitoli creati dalla comunità Amara.org", True),
        ("Sottotitoli a cura di QTSS", True),
        ("Thank you for watching!", True),
        ("Grazie per la visione.", True),
        ("...", True),
        ("Grazie mille, a domani!", False),
        ("Thank you for coming.", False),
    ],
)
def test_hallucination_filter(text, expected):
    assert is_hallucination(text) is expected


def test_choose_language_restricted_to_pair():
    probs = [("es", 0.55), ("it", 0.35), ("en", 0.05), ("pt", 0.05)]
    lang, conf = choose_language(probs, "es", 0.55, ["it", "en"])
    assert lang == "it"
    assert conf == pytest.approx(0.35 / 0.40)
    assert choose_language(None, "en", 0.9, ["it", "en"]) == ("en", 1.0)
