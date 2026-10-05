import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_sentences(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

def extract_entities(text: str):
    doc = nlp(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]

def extract_claims_and_evidence(sentences: list[str]):
    claims = []
    evidence = []
    premises = []
    conclusions = []
    
    for i, sent in enumerate(sentences):
        doc = nlp(sent)
        
        has_num = any(ent.label_ in ["PERCENT", "CARDINAL", "QUANTITY", "MONEY"] for ent in doc.ents)
        has_study = any(token.lemma_.lower() in ["study", "research", "report", "survey", "according"] for token in doc)
        
        conclusion_markers = {"therefore", "thus", "hence", "conclude", "should", "must"}
        is_conclusion = any(token.lemma_.lower() in conclusion_markers or token.text.lower() in conclusion_markers for token in doc)
        
        if is_conclusion:
            conclusions.append(sent)
            claims.append({"id": f"cl_{i}", "text": sent})
        else:
            premises.append(sent)
            
            if has_num or has_study:
                evidence.append({"id": f"ev_{i}", "text": sent, "type": "statistic" if has_num else "study"})
                
                import re
                parts = re.split(r',\s+', sent)
                if len(parts) > 1 and ("according" in parts[0].lower() or "study" in parts[0].lower()):
                    claim_text = ", ".join(parts[1:]).strip()
                    claims.append({"id": f"cl_{i}_sub", "text": claim_text})
                else:
                    claims.append({"id": f"cl_{i}", "text": sent})
            else:
                claims.append({"id": f"cl_{i}", "text": sent})
                
    return claims, evidence, premises, conclusions
