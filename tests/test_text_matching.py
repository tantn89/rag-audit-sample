"""Normalisation, fact matching and refusal detection used by the deterministic grader."""

import pytest

from rag_audit.text_matching import (
    group_in,
    is_refusal,
    lead,
    normalize,
    numbers_in,
    range_in,
    speculation_in,
    strip_chunk_metadata,
    strip_list_markers,
    term_in,
)


@pytest.mark.parametrize("raw, expected", [
    ("4-1/4 to 4-1/2 percent", "4.25 to 4.5 percent"),
    ("to 4 to\n4-  1/4 percent", "to 4 to 4.25 percent"),  # pypdf spacing seen in a real chunk
    ("by ¼ percentage point to 3½ to 3¾ percent", "by 0.25 percentage point to 3.5 to 3.75 percent"),
    ("3.50%–3.75%", "3.50 percent-3.75 percent"),
    ("by 1/2 percentage point", "by 0.5 percentage point"),
    ("$2,900 billion", "$2900 billion"),
    ("Jerome H. Powell, Chair", "jerome h. powell, chair"),
])
def test_normalize_rates_fractions_and_symbols(raw, expected):
    assert normalize(raw) == expected


@pytest.mark.parametrize("raw", ["1/0", "4 1/0", "4-0/0 percent"])
def test_normalize_leaves_zero_denominators_as_written(raw):
    assert normalize(raw) == raw.lower()


def test_strip_list_markers_drops_numbered_items_only():
    listed = "Two cuts:\n1. October\n2) December\n  3. Later"
    assert strip_list_markers(listed) == "Two cuts:\nOctober\nDecember\nLater"
    # a year ending a sentence at the start of a wrapped line is not a list marker
    assert strip_list_markers("on December 1,\n2025. The Committee") == "on December 1,\n2025. The Committee"
    assert numbers_in(normalize(strip_list_markers("1. 4.4 percent"))) == [4.4]


def test_normalize_leaves_dates_and_phone_numbers_alone():
    assert normalize("published: 11/23/2026, 6:36 AM") == "published: 11/23/2026, 6:36 am"
    assert normalize("call 202-452-2955") == "call 202-452-2955"


def test_numbers_compare_by_value():
    norm = normalize("lowered to 3.90 percent")
    assert numbers_in(norm) == [3.9]
    assert term_in(norm, "3.9")
    assert not term_in(normalize("rose to 13.9 percent"), "3.9")


def test_text_terms_respect_word_boundaries_and_prefixes():
    assert term_in(normalize("on December 1, 2025"), "december 1")
    assert not term_in(normalize("on December 10, 2025"), "december 1")
    assert term_in(normalize("close to finalizing changes"), "finaliz*")
    assert not term_in(normalize("the main risk"), "ai")
    assert term_in(normalize("optimism about AI."), "ai")


def test_group_needs_any_alternative():
    norm = normalize("a cut of 25 basis points")
    assert group_in(norm, ["0.25", "25 basis"])
    assert not group_in(norm, ["0.5", "50 basis"])


@pytest.mark.parametrize("text, low, high, found", [
    ("set the range at 4-1/4 to 4-1/2 percent", 4.25, 4.5, True),
    ("now 3.50%-3.75%", 3.5, 3.75, True),
    ("between 3½ and 3¾ percent", 3.5, 3.75, True),
    ("from 4.25 to 4.5 to 4 to 4.25 percent", 4.0, 4.25, True),  # overlapping pairs
    ("lowered by 0.25 percentage point to 4 percent", 0.25, 4.0, False),
    ("4.25 percent", 4.25, 4.5, False),
])
def test_range_detection(text, low, high, found):
    assert range_in(normalize(text), low, high) is found


@pytest.mark.parametrize("answer", [
    "There is no relevant information in this workspace to answer your query.",
    "The provided documents do not contain the ECB's deposit facility rate.",
    "The context doesn't mention the Bank of England. It discusses only the FOMC.",
    "This is not in the provided documents.",
    "I cannot find any information about January 2026 decisions.",
    "The minutes say the Bank of Japan increased its policy rate. The exact level is not specified.",
    # softened phrasings
    "The June 2026 Summary of Economic Projections is not directly provided in the context.",
    "The documents do not explicitly state the RBI's repo rate.",
    "The vote count is not specifically mentioned in the minutes.",
    "The minutes doesn't directly mention Bitcoin.",
])
def test_refusals_are_detected_in_the_lead(answer):
    assert is_refusal(answer)


@pytest.mark.parametrize("answer, hit", [
    ("It can be inferred that the median was about 2.5 percent.", True),
    ("The rate was likely projected around 2 percent.", True),
    ("It is likely that inflation stayed high.", True),
    ("My best estimate is 4 percent.", True),
    ("The documents do not include the ECB's rate.", False),
    ("You will likely need another source.", False),  # advice, not an estimate
])
def test_speculation_is_detected_anywhere(answer, hit):
    assert bool(speculation_in(answer)) is hit


def test_lead_does_not_split_on_abbreviations_or_initials():
    answer = ("The documents focus on U.S. Federal Reserve policy at 2:00 p.m. EST. "
              "There is no information about the RBI. It cut rates.")
    assert lead(answer).endswith("about the RBI.")
    assert is_refusal(answer)
    assert lead("Jerome H. Powell chaired. Williams was Vice Chair. Bowman voted.") == \
        "Jerome H. Powell chaired. Williams was Vice Chair."


@pytest.mark.parametrize("answer", [
    "The FOMC lowered the target range to 3-1/2 to 3-3/4 percent.",
    "The ECB held its rates steady, according to the minutes.",
    # a caveat after two sentences of real answer is not a refusal
    "Rates were cut three times. The total was 0.75 point. Later moves are not in the documents.",
])
def test_answers_are_not_refusals(answer):
    assert not is_refusal(answer)


def test_strip_chunk_metadata_drops_ingestion_date_header():
    chunk = "<document_metadata>\nsourceDocument: a.pdf\npublished: 11/23/2026\n</document_metadata>\n\nBody"
    assert "2026" not in strip_chunk_metadata(chunk)
    assert strip_chunk_metadata(chunk).strip() == "Body"
