from app.services.extractor import extract_sentences, extract_claims_and_evidence, extract_entities
from app.services.verifier import verify_evidence
from app.services.scorer import calculate_scores
from app.services.nli import get_nli_pipeline


def process_text(text: str):
    sentences = extract_sentences(text)
    claims, evidence, premises, conclusions = extract_claims_and_evidence(sentences)

    nli = get_nli_pipeline()

    verification_results = []
    relationships = []
    fallacies = []
    warnings = []

    for ev in evidence:
        v_res = verify_evidence(ev["text"])
        verification_results.append({
            "evidence_id":          ev["id"],
            "status":               v_res["status"],
            "details":              v_res["details"],
            "provider_results":     v_res.get("provider_results", []),
            "best_match":           v_res.get("best_match"),
            "source_match_score":   v_res.get("source_match_score", 0.0),
            "evidence_comparison":  v_res.get("evidence_comparison"),
            "numerical_mismatches": v_res.get("numerical_mismatches", []),
            "searched_providers":   v_res.get("searched_providers", []),
        })

        for cl in claims:
            rel = nli.analyze_support(cl["text"], ev["text"])
            relationships.append({
                "claim_id":     cl["id"],
                "evidence_id":  ev["id"],
                "support_level":rel["support_level"],
                "explanation":  rel["explanation"],
            })

    # Fallacy detection runs over the full argument text
    fallacies = nli.detect_fallacies(text)

    if not evidence:
        warnings.append("No evidence found to support claims.")

    scores, overall, assessment = calculate_scores(
        claims, evidence, verification_results, relationships, fallacies
    )

    # Build source list from best_match metadata where available
    sources = []
    for vr in verification_results:
        bm = vr.get("best_match")
        if bm and bm.get("title"):
            sources.append({
                "id":   f"src_{vr['evidence_id']}",
                "name": bm["title"][:120],
                "url":  bm.get("url") or bm.get("doi"),
            })
    if evidence and not sources:
        sources = [{"id": "s1", "name": "Cited source (unverified)", "url": None}]

    return {
        "claims":                    claims,
        "premises":                  premises,
        "conclusions":               conclusions,
        "evidence":                  evidence,
        "sources":                   sources,
        "verification_results":      verification_results,
        "detected_fallacies":        fallacies,
        "claim_evidence_relationships": relationships,
        "component_scores":          scores,
        "overall_score":             overall,
        "overall_assessment":        assessment,
        "warnings":                  warnings,
        "confidence_levels":         {"extraction": 0.85, "verification": 0.65},
    }
