"""
The main public-facing class of the package.

This is what a user actually imports and calls:

    from hallufirewall import Firewall
    fw = Firewall(...)
    result = fw.check(answer=llm_answer, context=retrieved_chunks)

Internally, it just wires together the pieces we already built and tested
separately: claim extraction -> hybrid verification -> decision policy.
"""

from typing import Callable

from .claim_extraction import extract_claims_rule_based, extract_claims_llm
from .verifiers.nli_direct import NLIVerifierDirect
from .verifiers.llm_judge import LLMJudgeVerifier
from .verifiers.hybrid import HybridVerifier
from .policy import decide
from .types import FirewallResult


class Firewall:
    def __init__(
        self,
        llm_client: Callable[[str], str],
        claim_extraction_mode: str = "llm",   # "llm" or "rule_based"
        confidence_threshold: float = 0.85,
        nli_model_name: str = "roberta-large-mnli",
    ):
        """
        llm_client: a function (prompt: str) -> response: str.
                    Used for BOTH LLM-based claim extraction (if selected)
                    AND the LLM-judge verifier. Same dependency-injection
                    pattern we used throughout the project.
        claim_extraction_mode: "llm" (smarter, costs a call) or
                    "rule_based" (free, but naive sentence splitting).
        confidence_threshold: passed straight into HybridVerifier -- below
                    this, NLI's verdict gets escalated to the LLM-judge.
        """
        self.llm_client = llm_client
        self.claim_extraction_mode = claim_extraction_mode

        nli_verifier = NLIVerifierDirect(model_name=nli_model_name)
        llm_judge_verifier = LLMJudgeVerifier(llm_client=llm_client)
        self.verifier = HybridVerifier(
            nli_verifier=nli_verifier,
            llm_judge_verifier=llm_judge_verifier,
            confidence_threshold=confidence_threshold,
        )

    def check(self, answer: str, context: list[str]) -> FirewallResult:
        """
        The main entry point. Takes an LLM-generated answer plus the
        retrieved context chunks it was supposedly grounded in, and
        returns a full FirewallResult with a verdict for the whole answer.
        """
        # Step 1: break the answer into individual checkable claims.
        if self.claim_extraction_mode == "llm":
            claims = extract_claims_llm(answer, self.llm_client)
        else:
            claims = extract_claims_rule_based(answer)

        # Step 2: verify EACH claim against the context, using our hybrid
        # verifier (NLI first, escalate to LLM-judge when unsure).
        claim_verdicts = [self.verifier.verify(claim, context) for claim in claims]

        # Step 3: turn the list of per-claim verdicts into one overall
        # decision for the whole answer.
        verdict, overall_confidence = decide(claim_verdicts)

        return FirewallResult(
            original_answer=answer,
            claim_verdicts=claim_verdicts,
            verdict=verdict,
            overall_confidence=overall_confidence,
        )
