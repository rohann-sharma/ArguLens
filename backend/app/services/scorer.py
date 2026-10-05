"""
Scoring configuration for ArguLens argument quality assessment.

All weights are defined here in one place so they can be adjusted
without touching the scoring logic.

COMPONENT WEIGHTS (must sum to 1.0)
------------------------------------
claim_clarity           0.10  - how clearly claims are stated
evidence_presence       0.15  - whether any evidence is present at all
evidence_specificity    0.10  - how specific/concrete the evidence is
evidence_relevance      0.15  - how relevant evidence is to claims
claim_evidence_support  0.15  - whether evidence actually supports the claim
reasoning_quality       0.15  - absence of logical fallacies
logical_consistency     0.10  - internal consistency of the argument
source_verification     0.10  - ability to locate/verify cited sources
"""

# Maximum achievable score per component (used to normalise to 0-100)
SCORE_CEILING = 100

# Base scores — applied before any analysis-driven adjustments
BASE_SCORES = {
    "claim_clarity":       80,   # assume moderate clarity by default
    "evidence_presence":   0,    # starts at 0; boosted by presence of evidence
    "evidence_specificity":0,    # starts at 0; boosted by specificity signals
    "evidence_relevance":  40,   # neutral start
    "claim_evidence_support": 40,
    "reasoning_quality":   80,   # start high; fallacies reduce it
    "logical_consistency": 90,
    "source_verification": 0,    # starts at 0; only boosted when source found
}

# Verification status → source_verification score mapping
VERIFICATION_SCORE_MAP = {
    "FOUND_SUPPORTED":           90,
    "FOUND_PARTIALLY_SUPPORTED": 55,
    "FOUND_BUT_UNCLEAR":         35,
    "FOUND_CONTRADICTED":        15,
    "NOT_FOUND":                 20,
    "NO_SOURCE_PROVIDED":        0,
    # Legacy mock statuses (kept for backward compat)
    "Verified":                  90,
    "Partially verified":        55,
    "UNVERIFIED":                20,
    "No source provided":        0,
}

# Per-fallacy deductions
FALLACY_REASONING_PENALTY  = 25   # subtracted from reasoning_quality per fallacy
FALLACY_CONSISTENCY_PENALTY = 15  # subtracted from logical_consistency per fallacy
MIN_SCORE = 10                     # floor for any component


def calculate_scores(claims, evidence, verification_results, relationships, fallacies):
    scores = dict(BASE_SCORES)

    # --- Evidence presence & specificity ---
    if evidence:
        scores["evidence_presence"] = 100
        # Specificity: reward numeric/statistical evidence
        has_stats = any(ev.get("type") == "statistic" for ev in evidence)
        scores["evidence_specificity"] = 85 if has_stats else 55
    else:
        scores["evidence_presence"] = 0
        scores["evidence_specificity"] = 0

    # --- Claim clarity ---
    scores["claim_clarity"] = 85 if claims else 20

    # --- Source verification ---
    if verification_results:
        # Take the best (highest) verification score across all evidence pieces
        best_ver_score = max(
            VERIFICATION_SCORE_MAP.get(v["status"], 20)
            for v in verification_results
        )
        scores["source_verification"] = best_ver_score

    # --- Claim–evidence relationships ---
    if relationships:
        total = len(relationships)
        supported   = sum(1 for r in relationships if r["support_level"] == "Supports")
        partial     = sum(1 for r in relationships if r["support_level"] == "Partially supports")
        support_ratio = (supported + 0.5 * partial) / total
        scores["claim_evidence_support"] = int(20 + 80 * support_ratio)
        scores["evidence_relevance"]     = int(30 + 70 * support_ratio)

    # --- Fallacy penalties ---
    if fallacies:
        n = len(fallacies)
        scores["reasoning_quality"]   = max(MIN_SCORE, 80 - n * FALLACY_REASONING_PENALTY)
        scores["logical_consistency"] = max(MIN_SCORE, 90 - n * FALLACY_CONSISTENCY_PENALTY)

    # --- Numerical mismatch warning (reduce source_verification further) ---
    has_num_mismatch = any(
        len(v.get("numerical_mismatches", [])) > 0 for v in verification_results
    )
    if has_num_mismatch:
        scores["source_verification"] = max(MIN_SCORE, scores["source_verification"] - 20)

    overall = int(sum(scores.values()) / len(scores))
    assessment = "Poor" if overall <= 40 else "Moderate" if overall <= 70 else "Strong"

    return scores, overall, assessment
