"""
Formal tests for the decision policy.
"""

from hallufirewall.policy import decide
from hallufirewall.types import Claim, ClaimVerdict, Label


def make_verdict(label, confidence):
    return ClaimVerdict(claim=Claim(text="dummy"), label=label, confidence=confidence, evidence_chunks=[])


def test_all_supported_high_confidence_passes():
    verdicts = [make_verdict(Label.SUPPORTED, 0.99), make_verdict(Label.SUPPORTED, 0.95)]
    verdict, confidence = decide(verdicts)
    assert verdict == "pass"


def test_any_contradiction_blocks():
    verdicts = [make_verdict(Label.SUPPORTED, 0.99), make_verdict(Label.CONTRADICTED, 0.90)]
    verdict, confidence = decide(verdicts)
    assert verdict == "block"


def test_any_unverifiable_abstains():
    verdicts = [make_verdict(Label.SUPPORTED, 0.99), make_verdict(Label.UNVERIFIABLE, 0.40)]
    verdict, confidence = decide(verdicts)
    assert verdict == "abstain"


def test_low_confidence_supported_flags():
    verdicts = [make_verdict(Label.SUPPORTED, 0.99), make_verdict(Label.SUPPORTED, 0.60)]
    verdict, confidence = decide(verdicts)
    assert verdict == "flag"


def test_block_takes_priority_over_abstain():
    verdicts = [make_verdict(Label.CONTRADICTED, 0.90), make_verdict(Label.UNVERIFIABLE, 0.30)]
    verdict, confidence = decide(verdicts)
    assert verdict == "block"


def test_empty_claims_list_abstains():
    verdict, confidence = decide([])
    assert verdict == "abstain"


def test_overall_confidence_is_the_minimum_not_average():
    verdicts = [make_verdict(Label.SUPPORTED, 0.99), make_verdict(Label.SUPPORTED, 0.60)]
    verdict, confidence = decide(verdicts)
    assert confidence == 0.60
