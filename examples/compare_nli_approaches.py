"""
Comparing the zero-shot NLI trick (nli.py) against the direct MNLI
approach (nli_direct.py), across a small set of test cases ranging
from easy to hard.
"""

from hallufirewall.verifiers.nli import NLIVerifier
from hallufirewall.verifiers.nli_direct import NLIVerifierDirect
from hallufirewall.types import Claim

test_cases = [
    {
        "name": "EASY - true",
        "claim": "The sky appears blue on a clear day.",
        "context": ["On clear days, the sky appears blue due to a phenomenon called Rayleigh scattering."],
    },
    {
        "name": "EASY - false",
        "claim": "Water boils at 50 degrees Celsius at sea level.",
        "context": ["Water boils at 100 degrees Celsius at standard atmospheric pressure at sea level."],
    },
    {
        "name": "HARD - true (similar entities)",
        "claim": "Paris is the capital of France.",
        "context": ["France is a country in Europe. Its capital city is Paris."],
    },
    {
        "name": "HARD - false (similar entities)",
        "claim": "Paris is the capital of Germany.",
        "context": ["France is a country in Europe. Its capital city is Paris."],
    },
    {
        "name": "HARD - false (numeric)",
        "claim": "The company's revenue was $10 million in 2022.",
        "context": ["The company reported revenue of $12 million in 2022, up from the previous year."],
    },
]

print("Loading both models (this takes a minute, only happens once)...\n")
zero_shot_verifier = NLIVerifier()
direct_verifier = NLIVerifierDirect()

print("\n" + "=" * 90)
for case in test_cases:
    claim = Claim(text=case["claim"])
    context = case["context"]

    zs_result = zero_shot_verifier.verify(claim, context)
    direct_result = direct_verifier.verify(claim, context)

    print(f"\n[{case['name']}]")
    print(f"  Claim: {case['claim']}")
    print(f"  Zero-shot NLI : label={zs_result.label.value:<13} confidence={zs_result.confidence:.2f}")
    print(f"  Direct MNLI   : label={direct_result.label.value:<13} confidence={direct_result.confidence:.2f}")
print("\n" + "=" * 90)
