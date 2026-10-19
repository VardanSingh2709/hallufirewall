"""
Testing direct MNLI against harder cases specifically designed to target
known NLI blind spots: multi-hop reasoning, implicit negation, long
context, and paraphrasing. Comparing against the LLM-judge (Gemini).
"""

from dotenv import load_dotenv
from google import genai

from hallufirewall.verifiers.nli_direct import NLIVerifierDirect
from hallufirewall.verifiers.llm_judge import LLMJudgeVerifier
from hallufirewall.types import Claim

load_dotenv()
client = genai.Client()


def gemini_llm_client(prompt: str) -> str:
    response = client.models.generate_content(model="gemini-flash-lite-latest", contents=prompt)
    return response.text


test_cases = [
    {
        "name": "MULTI-HOP (true, requires combining 2 chunks)",
        "claim": "The current CEO of Acme Corp joined the company in 2019.",
        "context": [
            "The CEO of Acme Corp is John Carter.",
            "John Carter joined Acme Corp in 2019 as a senior engineer before being promoted.",
        ],
    },
    {
        "name": "IMPLICIT NEGATION (false, requires inference)",
        "claim": "The drug is safe for all ages without needing to consult a doctor.",
        "context": [
            "Patients under 18, pregnant individuals, and those with liver conditions should consult a doctor before use."
        ],
    },
    {
        "name": "PARAPHRASE (true, different wording + implicit comparison)",
        "claim": "The car's range exceeds 250 miles per charge.",
        "context": [
            "The vehicle can travel up to 300 miles on a single full charge under standard driving conditions."
        ],
    },
    {
        "name": "LONG CONTEXT (false detail buried in a long paragraph)",
        "claim": "The clinical trial included 500 participants.",
        "context": [
            "The clinical trial was conducted over 18 months across multiple sites in North America and Europe. "
            "Researchers screened over 2,000 candidates before enrollment. After exclusions for pre-existing "
            "conditions and incomplete screening, the final study enrolled 350 participants who completed the "
            "full treatment protocol. Results were published in a peer-reviewed journal the following year."
        ],
    },
]

nli = NLIVerifierDirect()
judge = LLMJudgeVerifier(llm_client=gemini_llm_client)

print("\n" + "=" * 100)
for case in test_cases:
    claim = Claim(text=case["claim"])
    context = case["context"]

    nli_result = nli.verify(claim, context)
    judge_result = judge.verify(claim, context)

    print(f"\n[{case['name']}]")
    print(f"  Claim: {case['claim']}")
    print(f"  Direct MNLI : label={nli_result.label.value:<13} confidence={nli_result.confidence:.2f}")
    print(f"  LLM-judge   : label={judge_result.label.value:<13} confidence={judge_result.confidence:.2f}")
print("\n" + "=" * 100)
