"""
Regression tests for NLP extraction and analysis.

These tests verify REAL NLP behavior using spaCy + scikit-learn.
No mock logic is used.
"""

import pytest
from app.services.extractor import extract_sentences, extract_claims_and_evidence, extract_entities
from app.services.nli import get_nli_pipeline


# ---------------------------------------------------------------------------
# Sentence extraction
# ---------------------------------------------------------------------------

def test_extract_sentences_basic():
    text = "This is sentence one. And sentence two!"
    sents = extract_sentences(text)
    assert len(sents) == 2


def test_extract_sentences_multi():
    text = "Vaccines are safe. Studies confirm this. Policy should reflect the evidence."
    sents = extract_sentences(text)
    assert len(sents) == 3


# ---------------------------------------------------------------------------
# Numeric expression preservation (regression for the 20,000 / comma bug)
# ---------------------------------------------------------------------------

def test_claim_extraction_20000_not_corrupted():
    """20,000 must NOT be split on the comma."""
    text = (
        "According to a 2024 Harvard study of 20,000 teenagers, "
        "80% of teenagers who use Instagram experience depression. "
        "Therefore Instagram should be banned for people under 18."
    )
    sentences = extract_sentences(text)
    claims, evidence, premises, conclusions = extract_claims_and_evidence(sentences)
    claim_texts = [c["text"] for c in claims]

    # The bug would produce "000 teenagers, 80%..." — make sure that's gone
    assert not any(c.startswith("000") for c in claim_texts), \
        f"Number was split incorrectly. Got claims: {claim_texts}"

    # The sub-claim must cleanly contain the statistic
    assert any("80% of teenagers" in c for c in claim_texts), \
        f"Expected '80% of teenagers' in a claim. Got: {claim_texts}"

    # The conclusion must be present
    assert any("Instagram should be banned" in c for c in claim_texts), \
        f"Expected ban claim. Got: {claim_texts}"


def test_claim_extraction_1500_not_corrupted():
    """1,500 must NOT be split on the comma."""
    text = "According to a 2023 report, 1,500 people were affected."
    sentences = extract_sentences(text)
    claims, evidence, _, _ = extract_claims_and_evidence(sentences)
    claim_texts = [c["text"] for c in claims]

    assert not any(c.startswith("500") for c in claim_texts), \
        f"1,500 was split incorrectly. Got: {claim_texts}"
    assert any("1,500" in c for c in claim_texts), \
        f"Expected '1,500' preserved in claim. Got: {claim_texts}"


def test_claim_extraction_2_5_million():
    """2.5 million must be preserved and correctly extracted."""
    text = "According to a 2023 report, 2.5 million users quit Instagram."
    sentences = extract_sentences(text)
    claims, evidence, _, _ = extract_claims_and_evidence(sentences)
    claim_texts = [c["text"] for c in claims]

    assert any("2.5 million" in c for c in claim_texts), \
        f"Expected '2.5 million' in a claim. Got: {claim_texts}"


def test_claim_extraction_percentage_80_5():
    """80.5% must be preserved as evidence and appear in claims."""
    text = "The study shows 80.5% of respondents disagreed."
    sentences = extract_sentences(text)
    claims, evidence, _, _ = extract_claims_and_evidence(sentences)

    # This sentence has a number so it should be classified as evidence
    assert len(evidence) >= 1
    assert "80.5%" in evidence[0]["text"]

    claim_texts = [c["text"] for c in claims]
    assert any("80.5%" in c for c in claim_texts), \
        f"Expected '80.5%' in a claim. Got: {claim_texts}"


def test_claim_extraction_currency_10000():
    """$10,000 must be preserved intact."""
    text = "Researchers found that $10,000 was spent per patient."
    sentences = extract_sentences(text)
    claims, evidence, _, _ = extract_claims_and_evidence(sentences)

    assert len(evidence) >= 1
    assert "$10,000" in evidence[0]["text"]

    claim_texts = [c["text"] for c in claims]
    assert any("$10,000" in c for c in claim_texts), \
        f"Expected '$10,000' in a claim. Got: {claim_texts}"


def test_claim_extraction_citation_2024_study():
    """Attribution phrase 'According to the 2024 study' must not appear in the claim text."""
    text = "According to the 2024 study, handwashing reduces illness by 21%."
    sentences = extract_sentences(text)
    claims, evidence, _, _ = extract_claims_and_evidence(sentences)
    claim_texts = [c["text"] for c in claims]

    # The citation prefix must be stripped from the extracted claim
    assert not any(c.strip().startswith("According to") for c in claim_texts), \
        f"Attribution phrase leaked into claim. Got: {claim_texts}"
    assert any("21%" in c or "handwashing" in c for c in claim_texts), \
        f"Expected claim content. Got: {claim_texts}"


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------

def test_extract_entities_percent_and_org():
    text = "According to Harvard, 80% of people sleep."
    entities = extract_entities(text)
    labels = [e["label"] for e in entities]
    assert "PERCENT" in labels, f"Expected PERCENT entity. Got: {labels}"
    assert any(l in labels for l in ["ORG", "PERSON", "GPE"]), \
        f"Expected ORG/PERSON/GPE for Harvard. Got: {labels}"


def test_extract_entities_date_and_cardinal():
    text = "A 2024 study of 20,000 teenagers found results."
    entities = extract_entities(text)
    labels = [e["label"] for e in entities]
    # Expect a numeric or date entity
    assert any(l in labels for l in ["DATE", "CARDINAL", "QUANTITY"]), \
        f"Expected DATE/CARDINAL. Got: {labels}"


# ---------------------------------------------------------------------------
# Claim-evidence similarity (TF-IDF cosine)
# ---------------------------------------------------------------------------

def test_similarity_related_texts():
    nli = get_nli_pipeline()
    claim = "Handwashing prevents diseases."
    evidence = "Washing hands regularly is known to reduce the spread of many diseases."
    result = nli.analyze_support(claim, evidence)
    assert result["support_level"] in ["Supports", "Partially supports"], \
        f"Expected support for related texts. Got: {result}"


def test_similarity_unrelated_texts():
    nli = get_nli_pipeline()
    result = nli.analyze_support("The sky is blue.", "Cats eat mice.")
    assert result["support_level"] == "Unrelated", \
        f"Expected Unrelated for unrelated texts. Got: {result}"


def test_similarity_explanation_contains_cosine():
    nli = get_nli_pipeline()
    result = nli.analyze_support("Vaccines are safe.", "Vaccines reduce the rate of infection.")
    assert "TF-IDF Cosine" in result["explanation"]


# ---------------------------------------------------------------------------
# Fallacy detection
# ---------------------------------------------------------------------------

def test_fallacy_hasty_generalization():
    nli = get_nli_pipeline()
    fallacies = nli.detect_fallacies("All politicians are totally corrupt.")
    types = [f["type"] for f in fallacies]
    assert "hasty generalization" in types, f"Expected hasty generalization. Got: {types}"


def test_fallacy_appeal_to_authority():
    nli = get_nli_pipeline()
    fallacies = nli.detect_fallacies("Because the doctor says so, it must be true.")
    types = [f["type"] for f in fallacies]
    assert "appeal to authority" in types, f"Expected appeal to authority. Got: {types}"


def test_fallacy_ad_hominem():
    nli = get_nli_pipeline()
    fallacies = nli.detect_fallacies("You are an idiot so your argument is wrong.")
    types = [f["type"] for f in fallacies]
    assert "ad hominem" in types, f"Expected ad hominem. Got: {types}"


def test_fallacy_correlation_vs_causation_harvard():
    """
    The Harvard Instagram argument uses observational data (experience depression)
    to draw a policy conclusion (should be banned). This is a textual pattern
    consistent with a correlation-vs-causation leap.
    """
    nli = get_nli_pipeline()
    text = (
        "According to a 2024 Harvard study of 20,000 teenagers, "
        "80% of teenagers who use Instagram experience depression. "
        "Therefore Instagram should be banned for people under 18."
    )
    fallacies = nli.detect_fallacies(text)
    types = [f["type"] for f in fallacies]
    assert "correlation vs causation" in types, \
        f"Expected correlation vs causation for Harvard example. Got: {types}"


def test_fallacy_no_false_positive_for_clean_argument():
    """A well-reasoned statement should not trigger correlation-vs-causation."""
    nli = get_nli_pipeline()
    text = (
        "A randomized controlled trial showed that the vaccine directly causes "
        "an immune response, therefore vaccination is recommended."
    )
    fallacies = nli.detect_fallacies(text)
    types = [f["type"] for f in fallacies]
    assert "correlation vs causation" not in types, \
        f"Should not fire for controlled trial language. Got: {types}"


def test_fallacy_slippery_slope():
    nli = get_nli_pipeline()
    fallacies = nli.detect_fallacies("If we allow this, it will inevitably lead to chaos.")
    types = [f["type"] for f in fallacies]
    assert "slippery slope" in types, f"Expected slippery slope. Got: {types}"
