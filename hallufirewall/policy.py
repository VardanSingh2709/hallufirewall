"""
Decision policy: turns a LIST of ClaimVerdicts (one per claim in the answer)
into ONE overall decision for the entire answer.

This directly implements our "high-stakes accuracy" design goal:
false negatives (letting a hallucination through) are far worse than
false positives (blocking/flagging something that was actually fine).
"""

from .types import ClaimVerdict, Label


# Below this confidence, even a "supported" claim gets treated as shaky
# enough to downgrade the overall answer from PASS to FLAG.
FLAG_CONFIDENCE_THRESHOLD = 0.85


def decide(claim_verdicts: list[ClaimVerdict]) -> tuple[str, float]:
    """
    Applies our high-stakes decision policy to a list of claim verdicts.

    Returns: (verdict, overall_confidence)
        verdict is one of: "pass", "flag", "abstain", "block"
        overall_confidence is the WORST (minimum) confidence across all
        claims -- NOT the average. In high-stakes settings, the answer
        is only as trustworthy as its weakest claim, not its average claim.
    """
    if not claim_verdicts:
        # No claims were extracted at all -- nothing to check.
        # We treat this conservatively: abstain rather than blindly pass.
        return "abstain", 0.0

    # "Weakest link" confidence: the whole answer's trustworthiness is
    # capped by its least-confident claim.
    overall_confidence = min(v.confidence for v in claim_verdicts)

    # RULE 1 (highest priority): any contradiction blocks the whole answer.
    if any(v.label == Label.CONTRADICTED for v in claim_verdicts):
        return "block", overall_confidence

    # RULE 2: any unverifiable claim means we don't have enough evidence
    # to safely pass the answer -- route to a human instead of guessing.
    if any(v.label == Label.UNVERIFIABLE for v in claim_verdicts):
        return "abstain", overall_confidence

    # At this point, every claim is SUPPORTED. The only question left is
    # whether we're confident enough to fully trust it.
    # RULE 3: any claim with lower-than-threshold confidence -> flag for review.
    if any(v.confidence < FLAG_CONFIDENCE_THRESHOLD for v in claim_verdicts):
        return "flag", overall_confidence

    # RULE 4: every claim supported, every claim confident -> pass cleanly.
    return "pass", overall_confidence
