"""Deterministic answer matching: normalise text and numbers, match key facts, detect refusals.

FOMC documents write rates as "4-1/4", "4 1/4", "4¼" or "4.25"; answers add "%" and
"25 basis points". Everything is normalised to lowercase text with decimal numbers so a fact
check compares values, not spellings:

    normalize("lower the range by ¼ percentage point to 3½ to 3¾%")
    -> "lower the range by 0.25 percentage point to 3.5 to 3.75 percent"
"""

from __future__ import annotations

import re
import unicodedata

_UNICODE_FRACTIONS = {"¼": "1/4", "½": "1/2", "¾": "3/4", "⅛": "1/8", "⅜": "3/8", "⅝": "5/8",
                      "⅞": "7/8", "⅓": "1/3", "⅔": "2/3"}
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−"), "-")
_QUOTES = {ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"'}
_NUM = r"\d+(?:\.\d+)?"
# A whole number followed by a simple fraction: "4-1/4", "4- 1/4", "4 1/4" (from "4¼").
_MIXED = re.compile(r"(?<![\d./])(\d+)(?:\s*-\s*|\s+)(\d)\s*/\s*(\d)(?![\d/])")
_FRACTION = re.compile(r"(?<![\d./])(\d)\s*/\s*(\d)(?![\d/])")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_NUMBER = re.compile(rf"(?<![\d.]){_NUM}")
_RANGE = re.compile(rf"(?=(?<![\d.])({_NUM})\s*(?:percent)?\s*(?:to|-|and|through)\s*\$?({_NUM})"
                    rf"(?![\d.]))")
_METADATA = re.compile(r"<document_metadata>.*?</document_metadata>", re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
# Periods that do not end a sentence: dotted abbreviations (U.S., a.m., e.g.), single-letter
# initials (H. in "Jerome H. Powell") and a few titles.
_ABBREVIATION = re.compile(r"\b(?:[A-Za-z]\.){2,}|\b[A-Z]\.(?=\s)|\b(?:Mr|Mrs|Ms|Dr|St|No|vs)\.")
_DOT = "․"  # placeholder while splitting
# A numbered-list marker at the start of a line ("1. ", "2) "). Only 1-2 digits, so a year that
# ends a sentence at a line start ("2025. The Committee") is not taken for one.
_LIST_MARKER = re.compile(r"(?m)^[ \t]*\d{1,2}[.)][ \t]+")
# An optional softener between "not" and the verb: "is not directly provided".
_SOFT = r"(?:directly |explicitly |specifically |clearly |fully |actually )?"

# Phrases that decline to answer from the documents. Matched in the answer's lead only, so a
# caveat at the end of a real answer is not a refusal.
REFUSAL_PATTERNS = [re.compile(p) for p in (
    r"\bno relevant information\b",
    rf"\b(?:do|does|did) not {_SOFT}(?:contain|include|mention|provide|specify|state|cover|have|"
    r"address|discuss|indicate|say|reference|give|report|list|detail)",
    rf"\b(?:don't|doesn't|didn't) {_SOFT}(?:contain|include|mention|provide|specify|state|cover|"
    r"have|address|discuss|indicate|say|reference|give|report)",
    rf"\bnot {_SOFT}(?:available|specified|mentioned|provided|included|stated|found|covered|"
    r"addressed|discussed|given|reported|contained|detailed|present)\b",
    r"\b(?:cannot|can't|can not|unable to|not able to) (?:find|answer|determine|provide|confirm|"
    r"identify|locate|say)",
    r"\bno (?:information|details|data|mention|record|reference|specific)\b",
    r"\bthere is no (?:information|mention|data|record|reference|indication)",
    r"\bnot (?:in|found in|included in|covered by|part of|within) (?:the |these |this |any of the )?"
    r"(?:provided |given |available |supplied )?(?:documents?|context|sources?|excerpts?|texts?|"
    r"materials?|information|records?|workspace)",
    r"\boutside (?:of )?(?:the )?(?:scope|provided|available)",
    r"\binsufficient (?:information|context)\b",
    r"\bi (?:do not|don't) (?:know|have)\b",
)]

# Hedged estimates. For a question the documents cannot answer, an answer that declines and then
# guesses ("it can be inferred that ... was likely around 2 percent") still states invented
# content, so these fail a refusal case wherever they appear in the answer.
SPECULATION_PATTERNS = [re.compile(p) for p in (
    r"\bcan be (?:inferred|deduced|estimated)\b",
    r"\b(?:it is|it's) (?:likely|probable|reasonable to assume)\b",
    r"\b(?:was|were|is|are|would be|will be) (?:likely|probably|presumably)\b",
    r"\b(?:likely|probably|presumably) (?:around|about|near|close to|to be|projected)\b",
    r"\b(?:best|rough|educated) (?:estimate|guess)\b",
)]


def _fmt(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _mixed(m: re.Match) -> str:
    # "4 1/0" is not a fraction: leave it as written instead of dividing by zero
    return m[0] if int(m[3]) == 0 else _fmt(int(m[1]) + int(m[2]) / int(m[3]))


def _fraction(m: re.Match) -> str:
    return m[0] if int(m[2]) == 0 else _fmt(int(m[1]) / int(m[2]))


def normalize(text: str | None) -> str:
    """Lowercase text with unicode fractions, mixed fractions and "%" turned into decimals."""
    t = text or ""
    for ch, frac in _UNICODE_FRACTIONS.items():
        t = t.replace(ch, f" {frac}")
    t = unicodedata.normalize("NFKC", t).replace("⁄", "/")
    t = t.translate(_DASHES).translate(_QUOTES).lower()
    t = _THOUSANDS.sub("", t)
    t = t.replace("%", " percent")
    t = _MIXED.sub(_mixed, t)
    t = _FRACTION.sub(_fraction, t)
    return re.sub(r"\s+", " ", t).strip()


def strip_chunk_metadata(text: str | None) -> str:
    """Drop AnythingLLM's <document_metadata> header (it carries the ingestion date)."""
    return _METADATA.sub(" ", text or "")


def strip_list_markers(text: str | None) -> str:
    """Drop numbered-list markers so "3. On December 10" does not count as the number 3."""
    return _LIST_MARKER.sub("", text or "")


def speculation_in(text: str | None) -> list[str]:
    """Hedged-estimate phrases anywhere in the text (see SPECULATION_PATTERNS)."""
    norm = normalize(text)
    return [m[0] for p in SPECULATION_PATTERNS for m in [p.search(norm)] if m]


def numbers_in(norm: str) -> list[float]:
    return [float(n) for n in _NUMBER.findall(norm)]


def is_numeric_term(term: str) -> bool:
    return re.fullmatch(_NUM, term) is not None


def term_in(norm: str, term: str) -> bool:
    """One key fact in normalised text: numbers by value, "word*" by prefix, else whole words."""
    t = normalize(term)
    if is_numeric_term(t):
        return any(abs(n - float(t)) < 1e-9 for n in numbers_in(norm))
    if t.endswith("*"):
        return re.search(rf"(?<![a-z0-9]){re.escape(t[:-1])}", norm) is not None
    return re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", norm) is not None


def group_in(norm: str, alternatives: list[str]) -> bool:
    return any(term_in(norm, alt) for alt in alternatives)


def range_in(norm: str, low: float, high: float) -> bool:
    """A "low to high" range stated anywhere, e.g. "3.5 to 3.75 percent" or "3.50-3.75"."""
    return any(abs(float(a) - low) < 1e-9 and abs(float(b) - high) < 1e-9
               for a, b in _RANGE.findall(norm))


def lead(text: str | None, sentences: int = 2) -> str:
    """The first sentences; "U.S.", "a.m." and initials such as "Jerome H. Powell" do not end one."""
    guarded = _ABBREVIATION.sub(lambda m: m[0].replace(".", _DOT), (text or "").strip())
    parts = _SENTENCE_END.split(guarded)
    return " ".join(parts[:sentences]).replace(_DOT, ".")


def is_refusal(text: str | None, sentences: int = 2) -> bool:
    """The answer declines (fully or for the asked detail) within its first sentences."""
    head = normalize(lead(text, sentences))
    return any(p.search(head) for p in REFUSAL_PATTERNS)
