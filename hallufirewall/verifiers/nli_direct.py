"""
An alternative NLI verifier using the STANDARD technique: feeding
(premise, hypothesis) pairs directly into a model trained specifically
for MNLI classification, instead of the zero-shot-classification trick
we used in nli.py.

This should, in theory, give more decisive/better-calibrated scores,
since the model is being used exactly the way it was trained -- not
repurposed via a workaround.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from ..types import Claim, ClaimVerdict, Label


class NLIVerifierDirect:
    """
    Same .verify() interface as NLIVerifier, but using direct MNLI
    classification instead of the zero-shot-classification pipeline.
    """

    def __init__(self, model_name: str = "roberta-large-mnli"):
        print(f"Loading direct MNLI model '{model_name}'... (only happens once)")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # roberta-large-mnli's label order is fixed: 0=contradiction, 1=neutral, 2=entailment
        # (this ordering is specific to this exact model -- it's documented on its HF model card)
        self.id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
        print("Model loaded.")

    def _check_single_chunk(self, claim_text: str, chunk_text: str) -> dict:
        # Feed premise and hypothesis directly, exactly how this model was trained.
        inputs = self.tokenizer(chunk_text, claim_text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0]
        return {self.id2label[i]: probs[i].item() for i in range(len(probs))}

    def verify(self, claim: Claim, context_chunks: list[str]) -> ClaimVerdict:
        best_scores = None
        best_chunk = None

        for chunk in context_chunks:
            scores = self._check_single_chunk(claim.text, chunk)
            if best_scores is None or scores["entailment"] > best_scores["entailment"]:
                best_scores = scores
                best_chunk = chunk

        if best_scores is None:
            return ClaimVerdict(claim=claim, label=Label.UNVERIFIABLE, confidence=0.0,
                                 evidence_chunks=[], verifier_used="nli_direct")

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
            verifier_used="nli_direct",
        )
