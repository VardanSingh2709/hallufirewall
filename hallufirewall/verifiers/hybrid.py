"""
Hybrid verification: combines the fast NLI verifier with the stronger,
slower LLM-judge verifier using an escalation policy.

Design principle (from our "high-stakes accuracy" framing):
    Only trust NLI's verdict when it's VERY confident. Anything less than
    that gets escalated to the LLM-judge, since our priority is catching
    hallucinations (avoiding false negatives), not saving cost.
"""

from typing import Protocol
from .llm_judge import LLMJudgeVerifier
from ..types import Claim, ClaimVerdict, Label


class Verifier(Protocol):
    """
    A Protocol formally documents "duck typing": ANY object with a
    .verify(claim, context_chunks) -> ClaimVerdict method satisfies this
    type, regardless of which class it actually is (NLIVerifierDirect,
    NLIVerifierDirect, or anything else built later). This is more
    accurate than hinting a specific class, since HybridVerifier
    genuinely doesn't care which concrete verifier it's given.
    """
    def verify(self, claim: Claim, context_chunks: list[str]) -> ClaimVerdict:
        ...


class HybridVerifier:
    """
    Combines a fast first-pass verifier (typically NLI) with a stronger,
    slower LLM-judge verifier using an escalation policy. Both are typed
    as `Verifier` above -- either can be swapped for any class with a
    matching .verify() method.
    """

    def __init__(
        self,
        nli_verifier: Verifier,
        llm_judge_verifier: LLMJudgeVerifier,
        confidence_threshold: float = 0.85,
    ):
        self.nli_verifier = nli_verifier
        self.llm_judge_verifier = llm_judge_verifier
        # Anything BELOW this confidence gets escalated to the LLM-judge.
        # 0.85 is deliberately strict, reflecting our high-stakes framing.
        self.confidence_threshold = confidence_threshold

    def verify(self, claim: Claim, context_chunks: list[str]) -> ClaimVerdict:
        # Step 1: always run the fast NLI check first.
        nli_result = self.nli_verifier.verify(claim, context_chunks)

        # Step 2: decide whether to trust it, or escalate.
        if nli_result.confidence >= self.confidence_threshold:
            # NLI is confident enough -- trust it, skip the expensive LLM call.
            # We relabel verifier_used so downstream code/logs know HYBRID
            # made this decision (and that it was resolved cheaply).
            nli_result.verifier_used = "hybrid (nli, high-confidence)"
            return nli_result

        # Step 3: NLI wasn't confident enough -- escalate to the LLM-judge.
        llm_result = self.llm_judge_verifier.verify(claim, context_chunks)
        llm_result.verifier_used = "hybrid (escalated to llm_judge)"
        return llm_result
