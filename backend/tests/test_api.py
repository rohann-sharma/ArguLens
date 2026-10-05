"""
Integration tests for the ArguLens FastAPI endpoints and core services.

These tests use:
- FastAPI TestClient for endpoint tests
- unittest.mock.patch for isolating external API calls in verify_evidence
- Direct function calls for scorer unit tests

All external HTTP calls are mocked so tests run without internet access.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app
from app.services.verifier import verify_evidence, _cache
from app.services.scorer import calculate_scores

client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_verifier_cache():
    """Clear verifier in-memory cache before each test."""
    _cache.clear()
    yield
    _cache.clear()


# ---------------------------------------------------------------------------
# Endpoint: POST /api/analyze
# ---------------------------------------------------------------------------

def test_analyze_endpoint_returns_200():
    """Basic endpoint smoke test: valid input → 200 with required fields."""
    with patch("app.services.verifier.search_crossref", return_value=[]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        response = client.post("/api/analyze", json={
            "text": "According to the CDC, washing hands helps. Thus we should do it."
        })

    assert response.status_code == 200
    data = response.json()
    assert "claims" in data
    assert "overall_score" in data
    assert "verification_results" in data
    assert "detected_fallacies" in data
    assert "component_scores" in data


def test_analyze_endpoint_verification_result_structure():
    """verification_results must include the new rich fields from Phase 2."""
    with patch("app.services.verifier.search_crossref", return_value=[]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        response = client.post("/api/analyze", json={
            "text": "According to a 2024 Harvard study, 80% experience depression."
        })

    assert response.status_code == 200
    data = response.json()

    # Must have at least one verification result
    assert len(data["verification_results"]) >= 1
    vr = data["verification_results"][0]

    # New Phase 2 schema fields must all be present
    assert "evidence_id" in vr
    assert "status" in vr
    assert "details" in vr
    assert "provider_results" in vr
    assert "best_match" in vr          # may be null
    assert "source_match_score" in vr
    assert "evidence_comparison" in vr  # may be null
    assert "numerical_mismatches" in vr
    assert "searched_providers" in vr

    # Status must be one of the defined Phase 2 values
    valid_statuses = {
        "FOUND_SUPPORTED", "FOUND_PARTIALLY_SUPPORTED", "FOUND_CONTRADICTED",
        "FOUND_BUT_UNCLEAR", "NOT_FOUND", "NO_SOURCE_PROVIDED",
    }
    assert vr["status"] in valid_statuses, f"Unexpected status: {vr['status']}"


def test_analyze_endpoint_no_evidence_text():
    """Argument with no evidence should return a warning and empty evidence list."""
    with patch("app.services.verifier.search_crossref", return_value=[]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        response = client.post("/api/analyze", json={
            "text": "Pineapple on pizza is bad. Everyone knows this."
        })

    assert response.status_code == 200
    data = response.json()
    assert "warnings" in data
    assert isinstance(data["warnings"], list)


def test_analyze_endpoint_harvard_example_claim_extraction():
    """
    The Harvard example must produce correctly extracted claims.
    The '000 teenagers' extraction bug must NOT be present.
    """
    with patch("app.services.verifier.search_crossref", return_value=[]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        response = client.post("/api/analyze", json={
            "text": (
                "According to a 2024 Harvard study of 20,000 teenagers, "
                "80% of teenagers who use Instagram experience depression. "
                "Therefore Instagram should be banned for people under 18."
            )
        })

    assert response.status_code == 200
    data = response.json()

    claim_texts = [c["text"] for c in data["claims"]]

    # Regression: '000 teenagers' must not appear (comma-split bug)
    assert not any(c.strip().startswith("000") for c in claim_texts), \
        f"Numeric split bug returned. Claims: {claim_texts}"

    # The statistical claim must be present
    assert any("80% of teenagers" in c for c in claim_texts), \
        f"Expected '80% of teenagers' in a claim. Got: {claim_texts}"

    # The ban conclusion must be present
    assert any("banned" in c for c in claim_texts), \
        f"Expected ban claim. Got: {claim_texts}"


def test_analyze_endpoint_fallacy_detection_harvard():
    """The Harvard example should trigger correlation-vs-causation detection."""
    with patch("app.services.verifier.search_crossref", return_value=[]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        response = client.post("/api/analyze", json={
            "text": (
                "According to a 2024 Harvard study of 20,000 teenagers, "
                "80% of teenagers who use Instagram experience depression. "
                "Therefore Instagram should be banned for people under 18."
            )
        })

    assert response.status_code == 200
    data = response.json()
    fallacy_types = [f["type"] for f in data["detected_fallacies"]]
    assert "correlation vs causation" in fallacy_types, \
        f"Expected correlation vs causation fallacy. Got: {fallacy_types}"


# ---------------------------------------------------------------------------
# verify_evidence: status values align with Phase 2 schema
# ---------------------------------------------------------------------------

def test_verify_evidence_not_found_status():
    """When no provider returns results, status must be NOT_FOUND."""
    with patch("app.services.verifier.search_crossref", return_value=[]), \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        res = verify_evidence("According to a 2024 Harvard study of 20,000 teenagers, 80% experience depression.")

    assert res["status"] == "NOT_FOUND"
    # Must not claim source is false
    assert "does not mean" in res["details"].lower() or "fabricated" in res["details"].lower()


def test_verify_evidence_no_source_provided():
    """Text without citation signals → NO_SOURCE_PROVIDED (no API calls needed)."""
    with patch("app.services.verifier.search_crossref", return_value=[]) as mock_cr, \
         patch("app.services.verifier.search_openalex", return_value=[]), \
         patch("app.services.verifier.search_pubmed",   return_value=[]):
        res = verify_evidence("Instagram is popular among teenagers.")

    assert res["status"] == "NO_SOURCE_PROVIDED"
    # No providers should have been searched since there's no citation
    assert mock_cr.call_count == 0


def test_verify_evidence_provider_failure_does_not_crash():
    """All providers failing must not crash; must return NOT_FOUND."""
    with patch("app.services.verifier.search_crossref", side_effect=Exception("Timeout")), \
         patch("app.services.verifier.search_openalex", side_effect=Exception("503")), \
         patch("app.services.verifier.search_pubmed",   side_effect=Exception("Network error")):
        res = verify_evidence("According to a 2024 Harvard study, results were found.")

    assert res["status"] == "NOT_FOUND"
    assert len(res["provider_results"]) == 3
    for pr in res["provider_results"]:
        assert "unavailable" in pr["status"].lower()


# ---------------------------------------------------------------------------
# calculate_scores: Phase 2 status values are correctly scored
# ---------------------------------------------------------------------------

def test_calculate_scores_found_supported_boosts_verification():
    claims   = [{"id": "c1", "text": "Claim A"}]
    evidence = [{"id": "e1", "text": "Evidence text", "type": "statistic"}]
    v_results = [{"evidence_id": "e1", "status": "FOUND_SUPPORTED",
                  "details": "", "numerical_mismatches": []}]
    rels      = [{"support_level": "Supports"}]
    fallacies = []

    scores, overall, assessment = calculate_scores(claims, evidence, v_results, rels, fallacies)

    assert scores["source_verification"] == 90
    assert overall > 70
    assert assessment == "Strong"


def test_calculate_scores_not_found_gives_partial_credit():
    claims   = [{"id": "c1", "text": "Claim A"}]
    evidence = [{"id": "e1", "text": "Evidence", "type": "study"}]
    v_results = [{"evidence_id": "e1", "status": "NOT_FOUND",
                  "details": "", "numerical_mismatches": []}]
    rels      = [{"support_level": "Supports"}]
    fallacies = []

    scores, overall, assessment = calculate_scores(claims, evidence, v_results, rels, fallacies)

    # NOT_FOUND maps to 20 (partial credit, since absence ≠ false)
    assert scores["source_verification"] == 20


def test_calculate_scores_numerical_mismatch_reduces_verification():
    claims   = [{"id": "c1", "text": "Claim"}]
    evidence = [{"id": "e1", "text": "Evidence", "type": "statistic"}]
    # FOUND_SUPPORTED but with a numerical mismatch
    v_results = [{"evidence_id": "e1", "status": "FOUND_SUPPORTED",
                  "details": "", "numerical_mismatches": [20000.0]}]
    rels      = [{"support_level": "Supports"}]
    fallacies = []

    scores, overall, assessment = calculate_scores(claims, evidence, v_results, rels, fallacies)

    # Should be 90 (FOUND_SUPPORTED base) - 20 (mismatch penalty) = 70
    assert scores["source_verification"] == 70


def test_calculate_scores_fallacy_reduces_reasoning():
    claims   = [{"id": "c1", "text": "Claim"}]
    evidence = [{"id": "e1", "text": "Evidence", "type": "statistic"}]
    v_results = [{"evidence_id": "e1", "status": "NOT_FOUND",
                  "details": "", "numerical_mismatches": []}]
    rels      = [{"support_level": "Partially supports"}]
    fallacies = [{"type": "hasty generalization", "description": "..."}]

    scores, _, _ = calculate_scores(claims, evidence, v_results, rels, fallacies)

    # reasoning_quality = 80 - 1*25 = 55
    assert scores["reasoning_quality"] == 55
    # logical_consistency = 90 - 1*15 = 75
    assert scores["logical_consistency"] == 75


def test_calculate_scores_no_evidence_zeroes_presence():
    claims   = [{"id": "c1", "text": "Claim"}]
    evidence = []
    v_results = []
    rels      = []
    fallacies = []

    scores, _, _ = calculate_scores(claims, evidence, v_results, rels, fallacies)

    assert scores["evidence_presence"] == 0
    assert scores["evidence_specificity"] == 0
