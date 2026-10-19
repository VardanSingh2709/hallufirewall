"""
The full end-to-end test: using the actual Firewall class, with real
NLI + real Gemini, on a realistic multi-claim answer that contains
ONE hallucinated fact mixed in with correct ones.
"""

from dotenv import load_dotenv
from google import genai

from hallufirewall import Firewall

load_dotenv()
client = genai.Client()


def gemini_llm_client(prompt: str) -> str:
    response = client.models.generate_content(model="gemini-flash-lite-latest", contents=prompt)
    return response.text


fw = Firewall(llm_client=gemini_llm_client, claim_extraction_mode="llm")

answer = (
    "Paris is the capital of France. It has a population of about 50 million people. "
    "The city is located on the Seine river."
)

context = [
    "France is a country in Europe. Its capital city is Paris.",
    "Paris has a population of just over 2 million people within the city limits.",
    "Paris is built along the banks of the Seine river.",
]

result = fw.check(answer=answer, context=context)

print("OVERALL VERDICT:", result.verdict)
print("OVERALL CONFIDENCE:", result.overall_confidence)
print()
print("PER-CLAIM BREAKDOWN:")
for cv in result.claim_verdicts:
    print(f"  - \"{cv.claim.text}\"")
    print(f"      label={cv.label.value}, confidence={cv.confidence:.2f}, verifier={cv.verifier_used}")
