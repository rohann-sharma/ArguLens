from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class AnalyzeRequest(BaseModel):
    text: str

class Claim(BaseModel):
    id: str
    text: str

class Evidence(BaseModel):
    id: str
    text: str
    type: str  # "statistic", "study", "quote"

class Source(BaseModel):
    id: str
    name: str
    url: Optional[str] = None

class VerificationResult(BaseModel):
    evidence_id: str
    status: str  # FOUND_SUPPORTED | FOUND_PARTIALLY_SUPPORTED | FOUND_CONTRADICTED |
                 # FOUND_BUT_UNCLEAR | NOT_FOUND | NO_SOURCE_PROVIDED
    details: str
    provider_results: List[Dict[str, Any]] = []
    best_match: Optional[Dict[str, Any]] = None
    source_match_score: float = 0.0
    evidence_comparison: Optional[Dict[str, Any]] = None
    numerical_mismatches: List[float] = []
    searched_providers: List[str] = []

class Fallacy(BaseModel):
    type: str
    description: str

class ClaimEvidenceRelationship(BaseModel):
    claim_id: str
    evidence_id: str
    support_level: str  # Supports | Partially supports | Does not support | Unrelated
    explanation: str

class ComponentScores(BaseModel):
    claim_clarity: int
    evidence_presence: int
    evidence_specificity: int
    evidence_relevance: int
    claim_evidence_support: int
    reasoning_quality: int
    logical_consistency: int
    source_verification: int

class ClaimAssessment(BaseModel):
    claim: str
    support_status: str
    explanation: str
    reasoning_issue: Optional[str] = None
    confidence: float

class LlmReasoning(BaseModel):
    enabled: bool
    error: Optional[str] = None
    claim_assessments: List[ClaimAssessment] = []
    overall_reasoning_assessment: Optional[str] = None
    key_strengths: List[str] = []
    key_weaknesses: List[str] = []

class AnalyzeResponse(BaseModel):
    claims: List[Claim]
    premises: List[str]
    conclusions: List[str]
    evidence: List[Evidence]
    sources: List[Source]
    verification_results: List[VerificationResult]
    detected_fallacies: List[Fallacy]
    claim_evidence_relationships: List[ClaimEvidenceRelationship]
    component_scores: ComponentScores
    overall_score: int
    overall_assessment: str
    warnings: List[str]
    confidence_levels: Dict[str, float]
    llm_reasoning: LlmReasoning
