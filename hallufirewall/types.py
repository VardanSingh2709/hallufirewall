"""
This file defines the core data structures used throughout the firewall.

Nothing in this file "does" anything -- it just defines the SHAPE of the data
that flows between different parts of the system. Think of it as the
vocabulary that the rest of the project will speak.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Label(Enum):
    """
    The possible outcomes when checking ONE claim against the retrieved context.

    Using an Enum instead of plain strings like "supported" / "unsupported"
    prevents typos -- if you accidentally write Label.SUPPORT instead of
    Label.SUPPORTED, Python will immediately error out instead of silently
    giving you a wrong result later.
    """
    SUPPORTED = "supported"        # the claim is backed by the retrieved context
    CONTRADICTED = "contradicted"  # the retrieved context says the OPPOSITE
    UNVERIFIABLE = "unverifiable"  # not enough evidence either way


@dataclass
class Claim:
    """
    One atomic factual statement extracted from an LLM's answer.

    Example: if the LLM's answer is:
        "Paris is the capital of France and has a population of 2 million."
    this might get split into TWO claims:
        Claim(text="Paris is the capital of France")
        Claim(text="Paris has a population of 2 million")
    """
    text: str                     # the claim itself, as a standalone sentence
    source_answer_span: str = ""  # (optional) the exact substring in the original answer


@dataclass
class ClaimVerdict:
    """
    The result of checking ONE claim against the retrieved context chunks.
    """
    claim: Claim
    label: Label
    confidence: float                     # 0.0 to 1.0, how sure the verifier is
    evidence_chunks: list[str] = field(default_factory=list)  # which chunks supported/refuted it
    verifier_used: str = ""               # "nli", "llm_judge", or "hybrid" -- useful for debugging


@dataclass
class FirewallResult:
    """
    The final result returned to the user after checking an ENTIRE answer.
    This is what fw.check(...) will return.
    """
    original_answer: str
    claim_verdicts: list[ClaimVerdict]
    verdict: str                  # "pass", "flag", "abstain", or "block"
    overall_confidence: float
    safe_answer: Optional[str] = None   # the cleaned-up answer, if we rewrote it
