import urllib.request
import json
import time

time.sleep(3)

text = (
    "According to a 2024 Harvard study of 20,000 teenagers, "
    "80% of teenagers who use Instagram experience depression. "
    "Therefore Instagram should be banned for people under 18."
)

data = json.dumps({"text": text}).encode("utf-8")
req = urllib.request.Request(
    "http://localhost:8003/api/analyze",
    data=data,
    headers={"Content-Type": "application/json"},
)

try:
    resp_bytes = urllib.request.urlopen(req, timeout=30).read()
    result = json.loads(resp_bytes.decode("utf-8"))

    print("=" * 60)
    print("CLAIMS:")
    for c in result.get("claims", []):
        print(f"  [{c['id']}] {c['text']}")

    print("\nEVIDENCE:")
    for e in result.get("evidence", []):
        print(f"  [{e['id']}] ({e['type']}) {e['text'][:100]}")

    print("\nDETECTED FALLACIES:")
    for f in result.get("detected_fallacies", []):
        print(f"  - {f['type']}: {f['description'][:80]}")

    print("\nVERIFICATION RESULTS:")
    for vr in result.get("verification_results", []):
        print(f"  evidence_id:        {vr['evidence_id']}")
        print(f"  status:             {vr['status']}")
        print(f"  details:            {vr['details'][:120]}")
        print(f"  source_match_score: {vr['source_match_score']}")
        print(f"  searched_providers: {vr['searched_providers']}")
        if vr.get("best_match"):
            bm = vr["best_match"]
            print(f"  best_match.title:   {bm.get('title', '')[:80]}")
            print(f"  best_match.year:    {bm.get('year')}")
            print(f"  best_match.provider:{bm.get('provider')}")
        else:
            print(f"  best_match:         None")
        if vr.get("evidence_comparison"):
            print(f"  evidence_comparison:{vr['evidence_comparison']}")
        print(f"  numerical_mismatches:{vr.get('numerical_mismatches', [])}")
        print(f"  provider_results:   {vr.get('provider_results', [])}")

    print("\nCOMPONENT SCORES:")
    for k, v in result.get("component_scores", {}).items():
        print(f"  {k:<30} {v}")

    print(f"\nOVERALL SCORE:      {result.get('overall_score')}")
    print(f"OVERALL ASSESSMENT: {result.get('overall_assessment')}")
    print(f"WARNINGS:           {result.get('warnings', [])}")
    print("=" * 60)

except Exception as e:
    print(f"ERROR: {e}")
