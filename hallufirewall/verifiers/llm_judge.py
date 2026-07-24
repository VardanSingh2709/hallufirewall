"""
LLM-judge verification: the stronger, slower check.

Same dependency-injection pattern as our LLM-based claim extractor:
we accept a `llm_client` function (prompt in, response text out) so this
code works with ANY LLM provider, not just one specific API.
"""

import json
from typing import Callable
from ..types import Claim, ClaimVerdict, Label
from ..claim_extraction import strip_markdown_json_fence


JUDGE_PROMPT = """You are a strict fact-checker. You will be given a CLAIM and some
CONTEXT passages. Decide whether the CONTEXT supports the CLAIM.

Respond with ONLY a JSON object in this exact shape, nothing else -- no markdown,
no explanation outside the JSON:

{{
  "label": "supported" | "contradicted" | "unverifiable",
  "confidence": <a number between 0.0 and 1.0>,
  "reasoning": "<one short sentence explaining your decision>"
}}

Rules:
- "supported"     -> the context clearly confirms the claim
- "contradicted"  -> the context clearly states something that conflicts with the claim
- "unverifiable"  -> the context does not contain enough information either way

CLAIM:
{claim}

CONTEXT:
{context}

JSON response:"""


class LLMJudgeVerifier:
    """
    Wraps an LLM-based verification call behind a simple .verify() method,
    matching the same interface as the NLI verifier so the rest of our system
    can treat both verifiers interchangeably.
    """

    def __init__(self, llm_client: Callable[[str], str]):
        # We store the function itself, to be called later inside verify().
        self.llm_client = llm_client

    def verify(self, claim: Claim, context_chunks: list[str]) -> ClaimVerdict:
        if not context_chunks:
            return ClaimVerdict(
                claim=claim,
                label=Label.UNVERIFIABLE,
                confidence=0.0,
                evidence_chunks=[],
                verifier_used="llm_judge",
            )

        # We join all chunks together and let the LLM reason across all of
        # them at once -- unlike our NLI verifier, LLMs handle long,
        # multi-chunk context in one go reasonably well.
        combined_context = "\n---\n".join(context_chunks)
        prompt = JUDGE_PROMPT.format(claim=claim.text, context=combined_context)

        raw_response = self.llm_client(prompt)
        cleaned_response = strip_markdown_json_fence(raw_response)

        try:
            parsed = json.loads(cleaned_response)
            label_str = parsed["label"]
            confidence = float(parsed["confidence"])
        except (json.JSONDecodeError, KeyError, ValueError):
            # If the LLM gave us something we can't parse, we fail SAFE:
            # treat it as unverifiable rather than guessing, and give it
            # zero confidence so downstream logic knows this needs review.
            return ClaimVerdict(
                claim=claim,
                label=Label.UNVERIFIABLE,
                confidence=0.0,
                evidence_chunks=context_chunks,
                verifier_used="llm_judge",
            )

        label_map = {
            "supported": Label.SUPPORTED,
            "contradicted": Label.CONTRADICTED,
            "unverifiable": Label.UNVERIFIABLE,
        }

        return ClaimVerdict(
            claim=claim,
            label=label_map.get(label_str, Label.UNVERIFIABLE),
            confidence=confidence,
            evidence_chunks=context_chunks,
            verifier_used="llm_judge",
        )
