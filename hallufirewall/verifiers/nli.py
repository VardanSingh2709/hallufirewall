"""
NLI-based verification using HuggingFace's zero-shot-classification pipeline.

NOTE: This is our ORIGINAL, WEAKER baseline approach, kept for comparison
purposes (see examples/compare_nli_approaches.py). The production verifier
used by Firewall is nli_direct.py instead -- see that file's docstring for why.
"""

from transformers import pipeline
from ..types import Claim, ClaimVerdict, Label


class NLIVerifier:
    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        print(f"Loading NLI model '{model_name}'... (only happens once)")
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        print("Model loaded.")

    def _check_single_chunk(self, claim_text: str, chunk_text: str) -> dict:
        candidate_labels = ["entailment", "contradiction", "neutral"]
        hypothesis_template = f"Given the context: '{chunk_text}', this statement is {{}}."

        result = self.classifier(
            claim_text,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
        )
        return dict(zip(result["labels"], result["scores"]))

    def verify(self, claim: Claim, context_chunks: list[str]) -> ClaimVerdict:
        best_scores = None
        best_chunk = None

        for chunk in context_chunks:
            scores = self._check_single_chunk(claim.text, chunk)
            if best_scores is None or scores["entailment"] > best_scores["entailment"]:
                best_scores = scores
                best_chunk = chunk

        if best_scores is None:
            return ClaimVerdict(
                claim=claim, label=Label.UNVERIFIABLE, confidence=0.0,
                evidence_chunks=[], verifier_used="nli",
            )

        top_label = max(best_scores, key=best_scores.get)
        label_map = {
            "entailment": Label.SUPPORTED,
            "contradiction": Label.CONTRADICTED,
            "neutral": Label.UNVERIFIABLE,
        }

        return ClaimVerdict(
            claim=claim,
            label=label_map[top_label],
            confidence=best_scores[top_label],
            evidence_chunks=[best_chunk] if best_chunk else [],
            verifier_used="nli",
        )
