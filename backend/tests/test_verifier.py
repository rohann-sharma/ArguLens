"""
Tests for the real source verification pipeline.

All HTTP calls are mocked so these tests run offline without hitting
external APIs. Tests validate the verification LOGIC — scoring, status
determination, numerical mismatch detection, cache, provider fallback.
"""

import pytest
from unittest.mock import patch

from app.services.verifier import (
    verify_evidence,
    extract_citation_clues,
    score_candidate,
    compare_evidence_to_abstract,
    numerical_overlap_score,
    extract_numbers,
    _cache,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HARVARD_TEXT = (
    "According to a 2024 Harvard study of 20,000 teenagers, "
    "80% of teenagers who use Instagram experience depression."
)

GOOD_CANDIDATE = {
    "provider": "Crossref",
    "title": "Social media use and depression in adolescents: a longitudinal study of Instagram",
    "authors": ["Smith J", "Jones A"],
    "year": 2024,
    "doi": "10.1234/example.2024",
    "url": "https://doi.org/10.1234/example.2024",
    "journal": "Journal of Adolescent Health",
    "abstract": (
        "A longitudinal study of 20,000 teenagers found that 80% of Instagram users "
        "reported clinically significant depressive symptoms compared to non-users."
    ),
}

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear in-memory cache before every test."""
    _cache.clear()
    yield
    _cache.clear()


# ---------------------------------------------------------------------------
# Test 1: Source found with strong metadata match
# ---------------------------------------------------------------------------

def test_source_found_strong_match():
    with patch("app.services.verifier.search_crossref", return_value=[GOOD_CANDIDATE]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        result = verify_evidence(HARVARD_TEXT)

    assert result["status"] in {"FOUND_SUPPORTED", "FOUND_PARTIALLY_SUPPORTED"}
    assert result["best_match"] is not None
    assert result["best_match"]["title"] != ""
    assert result["source_match_score"] > 0.30
    assert "Crossref" in result["searched_providers"]
    # Abstract comparison must be populated
    assert result["evidence_comparison"] is not None
    assert result["evidence_comparison"]["similarity"] > 0.0


# ---------------------------------------------------------------------------
# Test 2: Source found but year mismatch
# ---------------------------------------------------------------------------

def test_source_found_year_mismatch():
    old = {**GOOD_CANDIDATE, "year": 2018}

    with patch("app.services.verifier.search_crossref", return_value=[old]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        result = verify_evidence(HARVARD_TEXT)

    bd = result["best_match"]["match_score_breakdown"]
    assert "year" in bd
    # The year breakdown should mention it's not an exact match
    assert "mismatch" in bd["year"].lower() or "off by" in bd["year"].lower()

    # Year mismatch alone shouldn't claim FOUND_SUPPORTED if score is also low
    assert result["source_match_score"] < 0.80  # score must be reduced


# ---------------------------------------------------------------------------
# Test 3: Source found but sample size mismatch
# ---------------------------------------------------------------------------

def test_source_found_sample_size_mismatch():
    small = {
        **GOOD_CANDIDATE,
        "abstract": (
            "A study of 200 teenagers (much smaller cohort) found some associations "
            "between Instagram use and mood, but results were not significant."
        ),
    }

    with patch("app.services.verifier.search_crossref", return_value=[small]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        result = verify_evidence(HARVARD_TEXT)

    # Claimed 20,000 vs source mentioning 200 → should flag numerical mismatch
    assert len(result["numerical_mismatches"]) > 0, (
        f"Expected numerical mismatches for 20,000 vs 200. Got: {result['numerical_mismatches']}"
    )


# ---------------------------------------------------------------------------
# Test 4: Source found with partial evidence support (weak abstract match)
# ---------------------------------------------------------------------------

def test_source_found_partial_support():
    partial = {
        **GOOD_CANDIDATE,
        "abstract": (
            "Adolescent social media usage patterns were examined. "
            "Some mood-related outcomes were observed in a subset of participants."
        ),
    }

    with patch("app.services.verifier.search_crossref", return_value=[partial]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        result = verify_evidence(HARVARD_TEXT)

    assert result["status"] in {
        "FOUND_SUPPORTED", "FOUND_PARTIALLY_SUPPORTED", "FOUND_BUT_UNCLEAR"
    }
    assert result["evidence_comparison"] is not None


# ---------------------------------------------------------------------------
# Test 5: Source found but abstract contradicts the evidence
# ---------------------------------------------------------------------------

def test_source_found_contradictory():
    contrary = {
        **GOOD_CANDIDATE,
        "title": "No evidence of depression link in Instagram users: a large cohort study",
        "abstract": (
            "After controlling for pre-existing mental health conditions, "
            "no statistically significant association was found between Instagram "
            "use and depression in adolescents."
        ),
    }

    with patch("app.services.verifier.search_crossref", return_value=[contrary]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        result = verify_evidence(HARVARD_TEXT)

    # Must find the source (title is related)
    assert result["best_match"] is not None
    # Status can be anything valid — key check: evidence_comparison is populated
    assert result["evidence_comparison"] is not None
    # Similarity to a contradictory abstract should be low
    assert result["evidence_comparison"]["similarity"] < 0.5


# ---------------------------------------------------------------------------
# Test 6: Source not found (all providers return empty)
# ---------------------------------------------------------------------------

def test_source_not_found():
    with patch("app.services.verifier.search_crossref", return_value=[]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        result = verify_evidence(HARVARD_TEXT)

    assert result["status"] == "NOT_FOUND"
    assert result["best_match"] is None
    assert result["source_match_score"] == 0.0
    # Must not claim the source is fake
    assert "does not mean" in result["details"].lower() or \
           "fabricated" in result["details"].lower() or \
           "does not exist" in result["details"].lower()


# ---------------------------------------------------------------------------
# Test 7: No source provided (text has no citation signal)
# ---------------------------------------------------------------------------

def test_no_source_provided():
    with patch("app.services.verifier.search_crossref", return_value=[]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        result = verify_evidence("Instagram is popular among teenagers.")

    assert result["status"] == "NO_SOURCE_PROVIDED"
    assert result["best_match"] is None
    assert result["searched_providers"] == []


# ---------------------------------------------------------------------------
# Test 8: All providers fail (network errors)
# ---------------------------------------------------------------------------

def test_all_providers_fail():
    with patch("app.services.verifier.search_crossref", side_effect=Exception("Network error")), \
         patch("app.services.verifier.search_openalex", side_effect=ConnectionError("Timeout")), \
         patch("app.services.verifier.search_pubmed",   side_effect=Exception("503 Service Unavailable")):
        # Must NOT raise — should degrade gracefully
        result = verify_evidence(HARVARD_TEXT)

    assert result["status"] == "NOT_FOUND"
    # All three providers must be listed in provider_results as unavailable
    provider_names = {pr["provider"] for pr in result["provider_results"]}
    assert "Crossref"  in provider_names
    assert "OpenAlex"  in provider_names
    assert "PubMed"    in provider_names
    for pr in result["provider_results"]:
        assert "unavailable" in pr["status"].lower()


# ---------------------------------------------------------------------------
# Test 9: Multiple providers — one fails, another succeeds
# ---------------------------------------------------------------------------

def test_one_provider_fails_another_succeeds():
    with patch("app.services.verifier.search_crossref", side_effect=Exception("Timeout")), \
         patch("app.services.verifier.search_openalex", return_value=[GOOD_CANDIDATE]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        result = verify_evidence(HARVARD_TEXT)

    # OpenAlex succeeded — should have a best_match
    assert result["best_match"] is not None
    assert result["status"] != "NOT_FOUND"
    # Crossref should be listed as unavailable
    crossref_entry = next(
        (pr for pr in result["provider_results"] if pr["provider"] == "Crossref"), None
    )
    assert crossref_entry is not None
    assert "unavailable" in crossref_entry["status"].lower()
    # OpenAlex should show success
    openalex_entry = next(
        (pr for pr in result["provider_results"] if pr["provider"] == "OpenAlex"), None
    )
    assert openalex_entry is not None
    assert "result" in openalex_entry["status"].lower()


# ---------------------------------------------------------------------------
# Test 10: Numerical mismatch detection (unit level)
# ---------------------------------------------------------------------------

def test_numerical_mismatch_large_difference():
    claimed  = [20000.0, 80.0]
    source   = [200.0, 15.0]
    score, mismatches = numerical_overlap_score(claimed, source)
    assert score < 0.5
    assert len(mismatches) > 0
    assert 20000.0 in mismatches


def test_numerical_match_within_tolerance():
    claimed = [20000.0]
    source  = [19800.0, 80.0]   # 19800 is within 20% of 20000
    score, mismatches = numerical_overlap_score(claimed, source)
    assert score >= 1.0
    assert mismatches == []


def test_numerical_no_claimed_values():
    score, mismatches = numerical_overlap_score([], [100.0, 200.0])
    assert score == 0.5   # neutral
    assert mismatches == []


def test_numerical_no_source_values():
    score, mismatches = numerical_overlap_score([20000.0], [])
    assert score == 0.0
    assert 20000.0 in mismatches


# ---------------------------------------------------------------------------
# Helper unit tests
# ---------------------------------------------------------------------------

def test_extract_citation_clues_year():
    clues = extract_citation_clues(HARVARD_TEXT)
    assert clues["year"] == 2024


def test_extract_citation_clues_org_harvard():
    clues = extract_citation_clues(HARVARD_TEXT)
    assert any("Harvard" in o for o in clues["orgs"])


def test_extract_citation_clues_sample_size_20000():
    clues = extract_citation_clues(HARVARD_TEXT)
    assert 20000.0 in clues["sample_sizes"], \
        f"Expected 20000.0 in sample_sizes. Got: {clues['sample_sizes']}"


def test_extract_citation_clues_percentage():
    clues = extract_citation_clues(HARVARD_TEXT)
    assert 80.0 in clues["percentages"], \
        f"Expected 80.0 in percentages. Got: {clues['percentages']}"


def test_extract_citation_clues_no_citation():
    clues = extract_citation_clues("Instagram is very popular with young people.")
    assert clues["has_citation"] is False


def test_extract_citation_clues_has_citation():
    clues = extract_citation_clues("According to a 2023 report, usage is rising.")
    assert clues["has_citation"] is True


def test_extract_numbers_basic():
    nums = extract_numbers("A study of 20,000 people found that 80% were affected, costing $10,000.")
    assert 20000.0 in nums
    assert 80.0 in nums
    assert 10000.0 in nums


def test_compare_evidence_similar_texts():
    result = compare_evidence_to_abstract(
        "Instagram use causes depression in teenagers",
        "A study found Instagram use is associated with depressive symptoms in adolescents"
    )
    assert result["similarity"] > 0.05
    assert result["level"] in {"partial", "strong"}


def test_compare_evidence_empty_abstract():
    result = compare_evidence_to_abstract("Something happened.", "")
    assert result["level"] == "insufficient_data"
    assert result["similarity"] == 0.0


def test_candidate_score_exact_year():
    clues = extract_citation_clues(HARVARD_TEXT)
    candidate = {**GOOD_CANDIDATE, "year": 2024}
    score, bd = score_candidate(candidate, clues)
    assert bd["year"] == "exact match"
    assert score > 0.25


def test_candidate_score_year_mismatch_reduces_score():
    clues = extract_citation_clues(HARVARD_TEXT)
    score_2024, _ = score_candidate({**GOOD_CANDIDATE, "year": 2024}, clues)
    score_2010, bd = score_candidate({**GOOD_CANDIDATE, "year": 2010}, clues)
    assert score_2024 > score_2010
    assert "mismatch" in bd["year"].lower() or "off by" in bd["year"].lower()


def test_candidate_score_org_match_boosts_score():
    clues = extract_citation_clues(HARVARD_TEXT)
    with_harvard    = {**GOOD_CANDIDATE, "abstract": "Published by Harvard researchers in 2024."}
    without_harvard = {**GOOD_CANDIDATE, "abstract": "Published by unknown institution."}
    score_with, bd_with       = score_candidate(with_harvard,    clues)
    score_without, bd_without = score_candidate(without_harvard, clues)
    assert bd_with["org"] != "no org/institution match"
    # Score with Harvard mention should be >= without
    assert score_with >= score_without


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------

def test_cache_prevents_duplicate_provider_calls():
    with patch("app.services.verifier.search_crossref", return_value=[]) as mock_cr, \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        verify_evidence(HARVARD_TEXT)
        verify_evidence(HARVARD_TEXT)   # second call — should use cache
    # Crossref should only have been called once
    assert mock_cr.call_count == 1


def test_cache_result_is_same_object():
    with patch("app.services.verifier.search_crossref", return_value=[GOOD_CANDIDATE]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        r1 = verify_evidence(HARVARD_TEXT)
        r2 = verify_evidence(HARVARD_TEXT)
    assert r1 is r2
