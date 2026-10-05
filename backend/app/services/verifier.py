"""
Real source verification pipeline for ArguLens.

Queries public scholarly APIs (Crossref, OpenAlex, PubMed) to locate
sources cited in argument text. Uses TF-IDF similarity + numerical
matching to score how well a found source supports the claimed evidence.

STATUS DEFINITIONS
------------------
FOUND_SUPPORTED         - Source located, abstract consistent with claimed evidence.
FOUND_PARTIALLY_SUPPORTED - Source located, partial match (year diff, number mismatch, weak abstract match).
FOUND_CONTRADICTED      - Source located, abstract appears to contradict the claim.
FOUND_BUT_UNCLEAR       - Source located, but no abstract or insufficient text to confirm.
NOT_FOUND               - No matching source located. Does NOT imply the source is fake.
NO_SOURCE_PROVIDED      - No citation clues detected in the evidence text.
"""

import re
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Shared spaCy NLP (lazy import to avoid circular dependency at module load)
# ---------------------------------------------------------------------------
_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        from app.services.extractor import nlp
        _nlp = nlp
    return _nlp

# ---------------------------------------------------------------------------
# In-memory session cache  (key = lowercased/stripped evidence text[:200])
# ---------------------------------------------------------------------------
_cache: dict = {}

def _cache_key(text: str) -> str:
    return text.strip().lower()[:200]

# ---------------------------------------------------------------------------
# Citation clue extraction
# ---------------------------------------------------------------------------

def extract_citation_clues(text: str) -> dict:
    """
    Extract structured citation clues from evidence text using spaCy NER
    and targeted regex patterns.
    """
    nlp = _get_nlp()
    doc = nlp(text)

    # Year
    year_match = re.search(r'\b(19|20)\d{2}\b', text)
    year = int(year_match.group()) if year_match else None

    # Organizations and geopolitical entities (e.g. "Harvard", "CDC", "WHO")
    orgs = [ent.text for ent in doc.ents if ent.label_ in ("ORG", "GPE")]

    # Named persons (potential authors)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

    # Sample sizes: numeric expressions followed by population words
    sample_pattern = re.findall(
        r'([\d,]+(?:\.\d+)?(?:\s+million)?)\s+'
        r'(?:million\s+)?(?:participants|patients|people|teenagers|adolescents|'
        r'adults|children|subjects|individuals|users|respondents)',
        text, re.IGNORECASE
    )
    sample_sizes = []
    for raw in sample_pattern:
        cleaned = raw.strip().replace(',', '')
        try:
            if 'million' in cleaned.lower():
                sample_sizes.append(float(cleaned.lower().replace('million', '').strip()) * 1_000_000)
            else:
                sample_sizes.append(float(cleaned))
        except ValueError:
            pass

    # Percentages
    percentages = [float(p) for p in re.findall(r'(\d+(?:\.\d+)?)\s*%', text)]

    # Keywords: multi-word noun chunks + orgs
    keywords = [chunk.text.lower() for chunk in doc.noun_chunks if len(chunk.text.split()) >= 2]
    keywords.extend([o.lower() for o in orgs])
    keywords = list(dict.fromkeys(keywords))[:12]  # deduplicate, keep order, cap at 12

    # Does the text contain a citation signal?
    has_citation = bool(re.search(
        r'\b(according to|study|research|report|survey|journal|published|found that|researchers)\b',
        text, re.IGNORECASE
    ))

    return {
        "year": year,
        "orgs": orgs,
        "persons": persons,
        "sample_sizes": sample_sizes,
        "percentages": percentages,
        "keywords": keywords,
        "has_citation": has_citation,
        "raw_text": text,
    }

# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def extract_numbers(text: str) -> list:
    """Extract all numeric values from a text string."""
    result = []
    for token in re.findall(r'[\d,]+(?:\.\d+)?', text):
        try:
            result.append(float(token.replace(',', '')))
        except ValueError:
            pass
    return result


def numerical_overlap_score(claimed: list, source: list) -> tuple:
    """
    Compare numeric values between claimed evidence and source text.
    Returns (score 0.0-1.0, list of claimed values NOT found in source).
    A value is considered matched if the source contains a number within 20%.
    """
    if not claimed:
        return 0.5, []  # neutral — no numbers to dispute
    if not source:
        return 0.0, list(claimed)  # numbers claimed but nothing to compare

    mismatches = []
    matches = 0
    for cn in claimed:
        matched = any(
            sn > 0 and abs(cn - sn) / max(cn, sn) < 0.20
            for sn in source
        )
        if matched:
            matches += 1
        else:
            mismatches.append(cn)

    return matches / len(claimed), mismatches

# ---------------------------------------------------------------------------
# Evidence ↔ abstract comparison (TF-IDF cosine)
# ---------------------------------------------------------------------------

def compare_evidence_to_abstract(claimed_text: str, abstract: str) -> dict:
    """
    Transparent TF-IDF cosine similarity between claimed evidence and
    a candidate source abstract. NOT an NLI model.
    """
    if not abstract or not abstract.strip() or not claimed_text:
        return {"similarity": 0.0, "level": "insufficient_data"}
    try:
        vec = TfidfVectorizer(stop_words='english')
        m = vec.fit_transform([claimed_text, abstract])
        sim = float(cosine_similarity(m[0:1], m[1:2])[0][0])
    except ValueError:
        sim = 0.0

    level = (
        "strong"  if sim > 0.30 else
        "partial" if sim > 0.10 else
        "weak"    if sim > 0.01 else
        "unrelated"
    )
    return {"similarity": round(sim, 3), "level": level}

# ---------------------------------------------------------------------------
# Provider: Crossref
# ---------------------------------------------------------------------------

def search_crossref(clues: dict, timeout: int = 8) -> list:
    """Query api.crossref.org/works. Returns list of candidate dicts."""
    if not clues["keywords"] and not clues["orgs"]:
        return []

    query = " ".join((clues["orgs"] + clues["keywords"])[:8])[:200]
    params: dict = {
        "query": query,
        "rows": 5,
        "select": "DOI,title,author,published,abstract,container-title,URL",
    }
    if clues["year"]:
        params["filter"] = (
            f"from-pub-date:{clues['year'] - 1},"
            f"until-pub-date:{clues['year'] + 1}"
        )

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get("https://api.crossref.org/works", params=params)
            resp.raise_for_status()
            items = resp.json().get("message", {}).get("items", [])
    except Exception:
        raise  # re-raise so caller can track provider failure

    results = []
    for item in items:
        title_raw = item.get("title", [""])
        title = title_raw[0] if isinstance(title_raw, list) and title_raw else str(title_raw)

        authors = []
        for a in item.get("author", [])[:3]:
            name = f"{a.get('given', '')} {a.get('family', '')}".strip()
            if name:
                authors.append(name)

        year = None
        date_parts = item.get("published", {}).get("date-parts", [[]])
        if date_parts and date_parts[0]:
            year = date_parts[0][0]

        results.append({
            "provider": "Crossref",
            "title": title,
            "authors": authors,
            "year": year,
            "doi": item.get("DOI"),
            "url": item.get("URL"),
            "journal": " ".join(item.get("container-title", [])),
            "abstract": item.get("abstract", ""),
        })
    return results

# ---------------------------------------------------------------------------
# Provider: OpenAlex
# ---------------------------------------------------------------------------

def search_openalex(clues: dict, timeout: int = 8) -> list:
    """Query api.openalex.org/works. Returns list of candidate dicts."""
    if not clues["keywords"] and not clues["orgs"]:
        return []

    query = " ".join((clues["orgs"] + clues["keywords"])[:8])[:200]
    params: dict = {
        "search": query,
        "per_page": 5,
        "select": "id,title,authorships,publication_year,doi,primary_location,abstract_inverted_index",
    }
    if clues["year"]:
        params["filter"] = f"publication_year:{clues['year'] - 1}-{clues['year'] + 1}"

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                "https://api.openalex.org/works",
                params=params,
                headers={"User-Agent": "ArguLens/1.0 (mailto:argulens@example.com)"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        raise

    results = []
    for item in data.get("results", []):
        authors = [
            a.get("author", {}).get("display_name", "")
            for a in item.get("authorships", [])[:3]
        ]
        authors = [a for a in authors if a]

        # Reconstruct abstract from inverted index
        abstract = ""
        inv_idx = item.get("abstract_inverted_index") or {}
        if inv_idx:
            try:
                word_pos = sorted(
                    [(w, p) for w, positions in inv_idx.items() for p in positions],
                    key=lambda x: x[1],
                )
                abstract = " ".join(w for w, _ in word_pos[:150])
            except Exception:
                abstract = ""

        doi_raw = item.get("doi") or ""
        doi = doi_raw.replace("https://doi.org/", "") if doi_raw else None

        loc = (item.get("primary_location") or {})
        source = (loc.get("source") or {})
        journal = source.get("display_name", "")

        results.append({
            "provider": "OpenAlex",
            "title": item.get("title", ""),
            "authors": authors,
            "year": item.get("publication_year"),
            "doi": doi,
            "url": doi_raw or None,
            "journal": journal,
            "abstract": abstract,
        })
    return results

# ---------------------------------------------------------------------------
# Provider: PubMed (NCBI E-utilities)
# ---------------------------------------------------------------------------

def search_pubmed(clues: dict, timeout: int = 8) -> list:
    """Query NCBI E-utilities (esearch + esummary). Returns list of candidate dicts."""
    if not clues["keywords"] and not clues["orgs"]:
        return []

    query_parts = (clues["orgs"] + clues["keywords"])[:5]
    if clues["year"]:
        query_parts.append(f"{clues['year']}[dp]")
    query = " AND ".join(query_parts)[:300]

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={"db": "pubmed", "term": query, "retmax": 5, "retmode": "json"},
            )
            resp.raise_for_status()
            ids = resp.json().get("esearchresult", {}).get("idlist", [])

        if not ids:
            return []

        with httpx.Client(timeout=timeout) as client:
            resp = client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            )
            resp.raise_for_status()
            summary_data = resp.json()
    except Exception:
        raise

    results = []
    for pmid, item in summary_data.get("result", {}).items():
        if pmid == "uids":
            continue
        authors = [a.get("name", "") for a in item.get("authors", [])[:3] if a.get("name")]
        year = None
        y_match = re.search(r'\b(19|20)\d{2}\b', item.get("pubdate", ""))
        if y_match:
            year = int(y_match.group())

        results.append({
            "provider": "PubMed",
            "title": item.get("title", ""),
            "authors": authors,
            "year": year,
            "doi": None,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            "journal": item.get("source", ""),
            "abstract": "",  # esummary does not return abstracts
        })
    return results

# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def score_candidate(candidate: dict, clues: dict) -> tuple:
    """
    Produce an explainable score (0.0–1.0) for how well a candidate source
    matches the citation clues extracted from the claimed evidence.

    Breakdown weights
    -----------------
    year match          0.25
    org/author match    0.20
    keyword similarity  0.30
    sample size match   0.25
    """
    score = 0.0
    breakdown = {}

    title_text = candidate.get("title") or ""
    abstract_text = candidate.get("abstract") or ""
    journal_text = candidate.get("journal") or ""
    full_source_text = f"{title_text} {abstract_text} {journal_text}".lower()

    # --- 1. Year match (weight 0.25) ---
    if clues["year"] and candidate.get("year"):
        diff = abs(clues["year"] - candidate["year"])
        if diff == 0:
            score += 0.25
            breakdown["year"] = "exact match"
        elif diff == 1:
            score += 0.15
            breakdown["year"] = f"off by 1 year (claimed {clues['year']}, found {candidate['year']})"
        elif diff == 2:
            score += 0.05
            breakdown["year"] = f"off by 2 years (claimed {clues['year']}, found {candidate['year']})"
        else:
            breakdown["year"] = f"year mismatch: claimed {clues['year']}, found {candidate['year']}"
    elif not clues["year"]:
        score += 0.12  # neutral — no year was claimed
        breakdown["year"] = "no year claimed (neutral)"
    else:
        breakdown["year"] = "source year unknown"
        score += 0.05

    # --- 2. Org / author name in source text (weight 0.20) ---
    org_matches = [o for o in clues["orgs"] if o.lower() in full_source_text]
    if org_matches:
        score += min(0.20, 0.10 * len(org_matches))
        breakdown["org"] = f"matched: {org_matches}"
    else:
        breakdown["org"] = "no org/institution match"

    # --- 3. Keyword / semantic similarity via TF-IDF (weight 0.30) ---
    if clues["raw_text"] and (title_text or abstract_text):
        try:
            vec = TfidfVectorizer(stop_words='english')
            m = vec.fit_transform([clues["raw_text"], f"{title_text} {abstract_text}"])
            kw_sim = float(cosine_similarity(m[0:1], m[1:2])[0][0])
        except Exception:
            kw_sim = 0.0
        score += 0.30 * kw_sim
        breakdown["keyword_similarity"] = round(kw_sim, 3)
    else:
        breakdown["keyword_similarity"] = 0.0

    # --- 4. Sample size / numerical match (weight 0.25) ---
    if clues["sample_sizes"]:
        source_nums = extract_numbers(f"{title_text} {abstract_text}")
        num_score, mismatches = numerical_overlap_score(clues["sample_sizes"], source_nums)
        score += 0.25 * num_score
        if mismatches:
            breakdown["sample_size"] = (
                f"mismatch: claimed {clues['sample_sizes']}, "
                f"values {mismatches} not found in source text"
            )
        else:
            breakdown["sample_size"] = f"match for claimed sizes {clues['sample_sizes']}"
    else:
        score += 0.12  # neutral
        breakdown["sample_size"] = "no sample size claimed (neutral)"

    breakdown["total"] = round(score, 3)
    return round(score, 3), breakdown

# ---------------------------------------------------------------------------
# Status determination
# ---------------------------------------------------------------------------

def determine_status(
    best_score: float,
    evidence_comparison: dict,
    numerical_mismatches: list,
) -> str:
    if best_score < 0.20:
        return "NOT_FOUND"

    ev_sim = evidence_comparison.get("similarity", 0.0)
    ev_level = evidence_comparison.get("level", "insufficient_data")

    if ev_level == "insufficient_data":
        if best_score >= 0.30:
            return "FOUND_BUT_UNCLEAR"
        return "NOT_FOUND"

    # Significant numerical mismatch downgrades support
    if numerical_mismatches and best_score >= 0.30:
        return "FOUND_PARTIALLY_SUPPORTED"

    if best_score >= 0.45:
        if ev_sim > 0.25:
            return "FOUND_SUPPORTED"
        elif ev_sim > 0.08:
            return "FOUND_PARTIALLY_SUPPORTED"
        else:
            return "FOUND_BUT_UNCLEAR"

    if best_score >= 0.25:
        return "FOUND_PARTIALLY_SUPPORTED"

    return "NOT_FOUND"

# ---------------------------------------------------------------------------
# Main entry point (synchronous, called from pipeline.py)
# ---------------------------------------------------------------------------

def verify_evidence(evidence_text: str) -> dict:
    """
    Attempt to locate a source cited in evidence_text via public scholarly APIs.

    Returns a structured dict with status, details, best_match, scores, etc.
    NOT_FOUND does NOT imply the source is false or fabricated.
    """
    cache_key = _cache_key(evidence_text)
    if cache_key in _cache:
        return _cache[cache_key]

    clues = extract_citation_clues(evidence_text)

    # ---- Early exit: no citation detected ----
    if not clues["has_citation"]:
        result = {
            "status": "NO_SOURCE_PROVIDED",
            "details": "No citation or source reference was detected in this evidence statement.",
            "provider_results": [],
            "best_match": None,
            "source_match_score": 0.0,
            "evidence_comparison": None,
            "numerical_mismatches": [],
            "searched_providers": [],
        }
        _cache[cache_key] = result
        return result

    # ---- Search all providers ----
    all_candidates = []
    provider_statuses = {}

    for name, fn in [
        ("Crossref", search_crossref),
        ("OpenAlex", search_openalex),
        ("PubMed",   search_pubmed),
    ]:
        try:
            candidates = fn(clues)
            all_candidates.extend(candidates)
            provider_statuses[name] = f"returned {len(candidates)} result(s)"
        except Exception as exc:
            provider_statuses[name] = f"unavailable ({type(exc).__name__})"

    provider_results = [{"provider": p, "status": s} for p, s in provider_statuses.items()]

    if not all_candidates:
        result = {
            "status": "NOT_FOUND",
            "details": (
                "No matching source was located in the searched scholarly databases "
                "(Crossref, OpenAlex, PubMed). "
                "This does not mean the source does not exist or is fabricated."
            ),
            "provider_results": provider_results,
            "best_match": None,
            "source_match_score": 0.0,
            "evidence_comparison": None,
            "numerical_mismatches": [],
            "searched_providers": list(provider_statuses.keys()),
        }
        _cache[cache_key] = result
        return result

    # ---- Score and rank candidates ----
    scored = sorted(
        [(score_candidate(c, clues), c) for c in all_candidates],
        key=lambda x: x[0][0],
        reverse=True,
    )
    (best_score, best_breakdown), best_candidate = scored[0]

    # ---- Evidence ↔ abstract comparison ----
    evidence_comparison = compare_evidence_to_abstract(
        evidence_text, best_candidate.get("abstract", "")
    )

    # ---- Numerical mismatch check (sample sizes + percentages vs source) ----
    claimed_nums = clues["sample_sizes"] + clues["percentages"]
    source_nums = extract_numbers(
        f"{best_candidate.get('title', '')} {best_candidate.get('abstract', '')}"
    )
    _, numerical_mismatches = numerical_overlap_score(claimed_nums, source_nums)

    # ---- Determine status ----
    status = determine_status(best_score, evidence_comparison, numerical_mismatches)

    # ---- Build human-readable details ----
    title_snippet = (best_candidate.get("title") or "")[:100]
    ev_sim = evidence_comparison.get("similarity", 0.0)

    if status == "NOT_FOUND":
        details = (
            "No matching source was located in the searched scholarly databases. "
            "This does not mean the source does not exist or is fabricated."
        )
    elif status == "FOUND_SUPPORTED":
        details = (
            f"A plausible matching source was found: \"{title_snippet}\". "
            f"The abstract content is consistent with the claimed evidence "
            f"(TF-IDF similarity={ev_sim:.2f})."
        )
    elif status == "FOUND_PARTIALLY_SUPPORTED":
        parts = [f"A candidate source was found (\"{title_snippet}\"), but the match is partial."]
        if numerical_mismatches:
            parts.append(
                f"Numerical mismatch: claimed values {numerical_mismatches} "
                f"were not found in the source text."
            )
        elif ev_sim < 0.15:
            parts.append(
                f"The abstract has low similarity to the claimed evidence "
                f"(TF-IDF similarity={ev_sim:.2f})."
            )
        details = " ".join(parts)
    elif status == "FOUND_CONTRADICTED":
        details = (
            f"A source was found (\"{title_snippet}\") but its content "
            f"appears to contradict the claimed evidence."
        )
    else:  # FOUND_BUT_UNCLEAR
        details = (
            f"A candidate source was found (\"{title_snippet}\"), "
            f"but the available metadata is insufficient to confirm or deny the claimed evidence "
            f"(no abstract or too little text returned)."
        )

    result = {
        "status": status,
        "details": details,
        "provider_results": provider_results,
        "best_match": {
            "title": best_candidate.get("title", ""),
            "authors": best_candidate.get("authors", []),
            "year": best_candidate.get("year"),
            "journal": best_candidate.get("journal", ""),
            "doi": best_candidate.get("doi"),
            "url": best_candidate.get("url"),
            "provider": best_candidate.get("provider", ""),
            "abstract_snippet": (best_candidate.get("abstract") or "")[:500],
            "match_score_breakdown": best_breakdown,
        },
        "source_match_score": best_score,
        "evidence_comparison": evidence_comparison,
        "numerical_mismatches": [float(n) for n in numerical_mismatches],
        "searched_providers": list(provider_statuses.keys()),
    }
    _cache[cache_key] = result
    return result
