from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class SimilarityAnalysisPipeline:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def analyze_support(self, claim: str, evidence: str):
        """
        Transparent TF-IDF cosine similarity-based support analysis.
        This is NOT a trained NLI classifier. It measures lexical overlap
        between claim and evidence texts to estimate how related they are.
        """
        try:
            tfidf_matrix = self.vectorizer.fit_transform([claim, evidence])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except ValueError:
            similarity = 0.0
            
        if similarity > 0.15:
            support_level = "Supports"
        elif similarity > 0.05:
            support_level = "Partially supports"
        else:
            support_level = "Unrelated"
            
        return {
            "support_level": support_level,
            "explanation": f"Similarity-based support analysis (TF-IDF Cosine = {similarity:.2f})."
        }
        
    def detect_fallacies(self, text: str):
        """
        Explainable rule-based NLP fallacy detection layer (MVP).
        Each rule is independently checkable and labelled with its rationale.
        Rules are conservative — they only fire when textual patterns
        genuinely support the fallacy type.
        """
        fallacies = []
        text_lower = text.lower()
        
        # ------------------------------------------------------------------
        # 1. Appeal to authority
        # Pattern: "because [authority]" or "[authority] says" without
        # independent evidence being cited.
        # ------------------------------------------------------------------
        if re.search(r'\b(because|says).*?(doctor|expert|harvard|scientist|authority|professor|researcher)\b', text_lower):
            fallacies.append({
                "type": "appeal to authority",
                "description": "Relies on an authority figure's word without presenting the underlying evidence or reasoning."
            })
            
        # ------------------------------------------------------------------
        # 2. Ad hominem
        # Pattern: insults directed at a person rather than their argument.
        # ------------------------------------------------------------------
        if re.search(r'\b(idiot|stupid|corrupt|moron|hypocrite|ignorant|fool)\b', text_lower):
            fallacies.append({
                "type": "ad hominem",
                "description": "Attacks the person's character rather than addressing the argument."
            })
            
        # ------------------------------------------------------------------
        # 3. Hasty generalization
        # Pattern: absolute quantifiers applied broadly.
        # ------------------------------------------------------------------
        if re.search(r'\b(all |every |always |never |nobody |everyone |no one )\b', text_lower):
            fallacies.append({
                "type": "hasty generalization",
                "description": "Uses absolute terms ('all', 'every', 'never') which often indicate overly broad unsupported generalizations."
            })
            
        # ------------------------------------------------------------------
        # 4. Correlation vs Causation
        # Fires when:
        #   (a) The text contains an explicit causal/conclusive connector
        #       (therefore, thus, so we should, this proves, etc.)
        #   AND
        #   (b) The evidence clause uses observational/descriptive language
        #       (experience, associated with, linked to, correlated, found that)
        #   AND NOT
        #   (c) The text contains language indicating a controlled causal
        #       mechanism (randomized trial, controlled study, mechanism, causes).
        # ------------------------------------------------------------------
        has_conclusion_marker = bool(re.search(
            r'\b(therefore|thus|so we should|hence|this proves|this shows|should be banned|should be restricted)\b',
            text_lower
        ))
        has_observational_language = bool(re.search(
            r'\b(experience|associated with|linked to|correlated|found that|reported|showed that|percent of)\b',
            text_lower
        ))
        has_causal_mechanism = bool(re.search(
            r'\b(randomized|controlled study|controlled trial|mechanism|directly causes)\b',
            text_lower
        ))
        if has_conclusion_marker and has_observational_language and not has_causal_mechanism:
            fallacies.append({
                "type": "correlation vs causation",
                "description": (
                    "A conclusion or policy recommendation is drawn from observational data. "
                    "The evidence describes what was observed (e.g., a statistical association) "
                    "but does not establish a causal mechanism, yet the conclusion is stated as though causation is proven."
                )
            })
                
        # ------------------------------------------------------------------
        # 5. Slippery slope
        # Pattern: explicit slippery-slope language.
        # ------------------------------------------------------------------
        if re.search(r'\b(if we allow|next thing you know|it will lead to|will inevitably|will eventually lead)\b', text_lower):
            fallacies.append({
                "type": "slippery slope",
                "description": "Suggests that a first step inevitably leads to a chain of negative events without demonstrating why."
            })

        return fallacies

nli_pipeline = SimilarityAnalysisPipeline()

def get_nli_pipeline():
    return nli_pipeline
