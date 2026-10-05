from app.services.extractor import extract_sentences, extract_claims_and_evidence

cases = [
    ("20,000 numeric preservation", "According to a 2024 Harvard study of 20,000 teenagers, 80% of teenagers who use Instagram experience depression."),
    ("1,500 numeric preservation", "According to a 2023 report, 1,500 people were affected."),
    ("2.5 million", "According to a 2023 report, 2.5 million users quit Instagram."),
    ("80.5 percent", "The study shows 80.5% of respondents disagreed."),
    ("currency $10,000", "Researchers found that $10,000 was spent per patient."),
    ("citation 2024 study", "According to the 2024 study, handwashing reduces illness by 21%."),
]

for label, text in cases:
    sents = extract_sentences(text)
    claims, evidence, premises, conclusions = extract_claims_and_evidence(sents)
    print(f"CASE: {label}")
    print(f"  CLAIMS: {[c['text'] for c in claims]}")
    print(f"  EVIDENCE: {[e['text'] for e in evidence]}")
    print()
