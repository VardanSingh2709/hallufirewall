"""
Testing the FULL hybrid pipeline with REAL components:
- Real NLI model (roberta-large-mnli, direct classification)
- Real LLM-judge (Gemini free tier)
"""

from dotenv import load_dotenv
from google import genai

from hallufirewall.verifiers.nli_direct import NLIVerifierDirect
from hallufirewall.verifiers.llm_judge import LLMJudgeVerifier
from hallufirewall.verifiers.hybrid import HybridVerifier
from hallufirewall.types import Claim

load_dotenv()
client = genai.Client()


def gemini_llm_client(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt,
    )
    return response.text


nli = NLIVerifierDirect()
llm_judge = LLMJudgeVerifier(llm_client=gemini_llm_client)
hybrid = HybridVerifier(nli_verifier=nli, llm_judge_verifier=llm_judge, confidence_threshold=0.85)

context = ["France is a country in Europe. Its capital city is Paris."]

print("=== Test 1: TRUE claim ===")
true_claim = Claim(text="Paris is the capital of France")
print(hybrid.verify(true_claim, context))

print()
print("=== Test 2: FALSE claim (the one NLI got wrong earlier) ===")
false_claim = Claim(text="Paris is the capital of Germany")
print(hybrid.verify(false_claim, context))
