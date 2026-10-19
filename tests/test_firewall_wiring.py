"""
Formal test for the Firewall class's orchestration logic, using mocking
so this test runs FAST and WITHOUT needing real models or network access.
"""

import json
from unittest.mock import patch, MagicMock

from hallufirewall.types import Claim, ClaimVerdict, Label


def test_firewall_end_to_end_wiring_with_mocks():
    with patch("hallufirewall.firewall.NLIVerifierDirect") as MockNLI, \
         patch("hallufirewall.firewall.LLMJudgeVerifier") as MockJudge:

        mock_nli_instance = MagicMock()
        mock_nli_instance.verify.return_value = ClaimVerdict(
            claim=Claim(text="dummy"), label=Label.SUPPORTED, confidence=0.99, evidence_chunks=[]
        )
        MockNLI.return_value = mock_nli_instance

        from hallufirewall import Firewall

        def fake_llm_client(prompt):
            return json.dumps(["Paris is the capital of France"])

        fw = Firewall(llm_client=fake_llm_client, claim_extraction_mode="llm")
        result = fw.check(answer="Paris is the capital of France.", context=["France... Paris..."])

        assert result.verdict == "pass"
        assert len(result.claim_verdicts) == 1
