"""
Evaluating the Firewall on a real benchmark: a balanced 40-example sample
from RAGTruth's QA test split (20 known-hallucinated, 20 known-clean).
"""

import json
import time
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError

from hallufirewall import Firewall

load_dotenv()
client = genai.Client()


def gemini_llm_client(prompt: str, max_retries: int = 5) -> str:
    wait_seconds = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model="gemini-flash-lite-latest", contents=prompt)
            time.sleep(13)
            return response.text
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                print(f"    (rate limited, waiting {wait_seconds}s before retry {attempt + 1}/{max_retries})")
                time.sleep(wait_seconds)
                wait_seconds *= 2
            else:
                raise

    raise RuntimeError("Exceeded max retries due to rate limiting.")


with open("data/ragtruth_qa_eval_sample.json") as f:
    dataset = json.load(f)

fw = Firewall(llm_client=gemini_llm_client, claim_extraction_mode="llm")

true_positive = 0
false_negative = 0
true_negative = 0
false_positive = 0

results_log = []

for i, example in enumerate(dataset):
    print(f"[{i + 1}/{len(dataset)}] Checking: \"{example['question'][:50]}...\"")

    result = fw.check(answer=example["answer"], context=example["context"])

    system_flagged_problem = result.verdict != "pass"
    actually_hallucinated = example["is_hallucinated"]

    if actually_hallucinated and system_flagged_problem:
        true_positive += 1
        outcome = "TP (caught it)"
    elif actually_hallucinated and not system_flagged_problem:
        false_negative += 1
        outcome = "FN (MISSED IT -- dangerous)"
    elif not actually_hallucinated and not system_flagged_problem:
        true_negative += 1
        outcome = "TN (correctly passed)"
    else:
        false_positive += 1
        outcome = "FP (false alarm)"

    print(f"    verdict={result.verdict}, ground_truth_hallucinated={actually_hallucinated} -> {outcome}")
    results_log.append({**example, "predicted_verdict": result.verdict, "outcome": outcome})

recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0
precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print(f"  True Positives  (caught hallucinations):  {true_positive}")
print(f"  False Negatives (MISSED hallucinations):  {false_negative}")
print(f"  True Negatives  (correctly passed):        {true_negative}")
print(f"  False Positives (false alarms):            {false_positive}")
print()
print(f"  RECALL on hallucinations:    {recall:.2%}   <-- most important for high-stakes")
print(f"  PRECISION:                   {precision:.2%}")
print(f"  F1 score:                     {f1:.2%}")
print("=" * 60)

with open("data/eval_results_log.json", "w") as f:
    json.dump(results_log, f, indent=2)
print("\nFull results saved to data/eval_results_log.json")
