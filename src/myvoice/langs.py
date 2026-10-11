"""Language table, translation-direction logic and a tiny text language guesser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Language:
    code: str          # ISO 639-1, as used by Whisper ("it")
    name: str          # English name, used in prompts ("Italian")
    region_code: str   # locale-style code, used by TranslateGemma ("it-IT")


LANGUAGES: dict[str, Language] = {
    lang.code: lang
    for lang in (
        Language("it", "Italian", "it-IT"),
        Language("en", "English", "en-US"),
        # Not used by default; listed so the pair is easy to change (also needs a Piper voice).
        Language("es", "Spanish", "es-ES"),
        Language("fr", "French", "fr-FR"),
        Language("de", "German", "de-DE"),
        Language("pt", "Portuguese", "pt-PT"),
    )
}


def get_language(code: str) -> Language:
    try:
        return LANGUAGES[code]
    except KeyError:
        raise ValueError(f"unsupported language {code!r} (known: {', '.join(LANGUAGES)})") from None


_DIRECTION_RE = re.compile(r"^\s*([a-z]{2})\s*(?:-|->|2|→|_to_)\s*([a-z]{2})\s*$")


def parse_direction(direction: str, languages: Sequence[str]) -> tuple[str, str] | None:
    """Return ``(src, tgt)`` for a fixed direction, or ``None`` for automatic detection."""
    if direction.strip().lower() == "auto":
        return None
    m = _DIRECTION_RE.match(direction.lower())
    if not m:
        raise ValueError(f"direction must be 'auto' or like 'it-en', got {direction!r}")
    src, tgt = m.groups()
    if {src, tgt} != set(languages):
        raise ValueError(f"direction {direction!r} does not match the language pair {list(languages)}")
    return src, tgt


def other_language(src: str, languages: Sequence[str]) -> str:
    a, b = languages
    if src == a:
        return b
    if src == b:
        return a
    raise ValueError(f"{src!r} is not in the language pair {list(languages)}")


# Very small stop-word lists: enough to tell the languages of the pair apart in typed text.
_STOPWORDS = {
    "it": set(
        "il lo la i gli le un una uno di da in con su per tra fra che non è e ed sono sei siamo "
        "ho hai ha abbiamo avete hanno mi ti si ci vi del della dei delle al alla nel nella "
        "questo questa quello quella come cosa perché quando dove anche più molto grazie ciao "
        "buongiorno buonasera sì io tu lui lei noi voi loro mio mia tuo tua vorrei posso".split()
    ),
    "en": set(
        "the a an of to in on at for with from by and or but not is are was were be been "
        "have has had do does did i you he she we they it this that these those what which "
        "who how why when where my your our their can could would should will please thanks "
        "thank hello hi yes no".split()
    ),
    "es": set("el la los las un una de del que y en es son por para con no se lo le muy gracias hola".split()),
    "fr": set("le la les un une des de du que et en est sont pour avec pas ne je tu il nous vous merci bonjour".split()),
    "de": set("der die das ein eine und ist sind nicht ich du er sie wir ihr mit für von zu danke hallo".split()),
    "pt": set("o a os as um uma de do da que e em é são para com não se eu você obrigado olá".split()),
}

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def guess_text_language(text: str, candidates: Sequence[str]) -> str:
    """Guess which of ``candidates`` a typed sentence is written in (stop-word vote)."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    # Split Italian elisions: "l'acqua" -> "l", "acqua"; "dell'anno" -> "dell", "anno".
    tokens: list[str] = []
    for w in words:
        tokens.extend(t for t in w.split("'") if t)
    scores = {c: sum(t in _STOPWORDS.get(c, ()) for t in tokens) for c in candidates}
    # Accented vowels are a strong hint for Italian vs English.
    if "it" in scores:
        scores["it"] += 2 * len(re.findall(r"[àèéìòù]", text.lower()))
    best = max(candidates, key=lambda c: (scores[c], -list(candidates).index(c)))
    return best
