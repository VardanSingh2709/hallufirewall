# hallufirewall

A hallucination-detection firewall for RAG (Retrieval-Augmented Generation) systems.

Sits between your RAG pipeline's answer generation and the final output shown to the user.
Checks whether the generated answer is actually grounded in the retrieved context, and
returns a verdict: **pass**, **flag**, **abstain**, or **block**.

Built for high-stakes use cases (e.g. medical, legal) where missing a hallucination
is far more costly than a false alarm.

## How it works

1. **Claim extraction** — breaks the answer into individual, checkable factual claims
2. **Hybrid verification** — checks each claim against the retrieved context using a fast
   local NLI model first; escalates to an LLM-judge only when the NLI verifier isn't
   confident enough (configurable threshold)
3. **Decision policy** — combines all per-claim verdicts into one overall answer-level decision

## Installation

```bash
git clone <your-repo-url>
cd hallufirewall
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -e .
```

Alternatively, install just the dependencies directly:
```bash
pip install -r requirements.txt
```

## Usage

```python
from hallufirewall import Firewall

def my_llm_client(prompt: str) -> str:
    # Wire this up to any LLM provider -- Anthropic, OpenAI, Gemini, etc.
    # Must return the model's raw text response.
    ...

fw = Firewall(llm_client=my_llm_client)

result = fw.check(
    answer="Paris is the capital of France. It has a population of 50 million.",
    context=[
        "France is a country in Europe. Its capital city is Paris.",
        "Paris has a population of just over 2 million people.",
    ],
)

print(result.verdict)              # "block" -- the population claim is contradicted
print(result.overall_confidence)
for cv in result.claim_verdicts:
    print(cv.claim.text, "->", cv.label, cv.confidence)
```

## Evaluation

Evaluated on a balanced 40-example sample from RAGTruth's QA test split:

| Metric | Value |
|---|---|
| Recall (hallucinations caught) | 90% |
| Precision | 64.3% |
| F1 | 75% |

## Project structure

```
hallufirewall/
    types.py               # core data structures (Claim, ClaimVerdict, FirewallResult, Label)
    claim_extraction.py     # rule-based + LLM-based claim extraction
    policy.py                # decision policy (pass/flag/abstain/block)
    firewall.py               # the main public Firewall class
    verifiers/
        nli_direct.py            # direct MNLI verifier
        llm_judge.py               # LLM-as-judge verifier
        hybrid.py                   # combines NLI + LLM-judge with escalation
```
