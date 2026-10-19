"""
Testing the LLM-judge verifier using Google's Gemini free tier as the backend.
"""

import os
from dotenv import load_dotenv
from google import genai
from hallufirewall.verifiers.llm_judge import LLMJudgeVerifier
from hallufirewall.types import Claim

load_dotenv()
client = genai.Client()


def gemini_llm_client(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-flash-lite-latest",
        contents=prompt,
    )
    return response.text


judge = LLMJudgeVerifier(llm_client=gemini_llm_client)

false_claim = Claim(text="Paris is the capital of Germany")
context = ["France is a country in Europe. Its capital city is Paris."]

result = judge.verify(false_claim, context)
print(result)
