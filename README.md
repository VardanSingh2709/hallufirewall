# hallufirewall

A hallucination-detection firewall for RAG (Retrieval-Augmented Generation) systems.

It sits between your RAG pipeline's answer generation and the user. Given an LLM-generated
answer and the context that was retrieved for it, it checks whether the answer is actually
grounded in that context and returns one verdict: **pass**, **flag**, **abstain** or **block**.

Built as a B.Tech final-year project, with high-stakes domains (medical, legal) in mind,
where missing a hallucination costs far more than raising a false alarm.

## How it works

```
Answer + Retrieved Context
        |
        v
1. Claim extraction     LLM splits the answer into atomic, checkable claims
        |
        v
2. Hybrid verification  per claim: local NLI model first (roberta-large-mnli);
                        if its confidence is below the threshold (default 0.85),
                        escalate that claim to an LLM-judge
        |
        v
3. Decision policy      combine all claim verdicts into one answer-level verdict
        |
        v
FirewallResult(verdict, overall_confidence, claim_verdicts)
```

### Verdicts

Checked in priority order, first match wins:

| Verdict | Triggered when | Meaning |
|---|---|---|
| **block** | any claim is contradicted by the context | The answer contains something wrong. Don't show it. |
| **abstain** | any claim is unverifiable (no evidence either way) | Route to a human reviewer. |
| **flag** | all claims supported, but at least one below the confidence threshold | Probably fine, worth a second look. |
| **pass** | all claims supported with high confidence | Safe to show. |

`overall_confidence` is the **minimum** confidence across claims, not the average, so one
weak claim cannot be hidden by strong ones.

### Design notes

- **Provider-agnostic LLM calls.** The package never imports an LLM SDK. You pass in
  `llm_client`, any function with the signature `(prompt: str) -> str`.
- **Claim-level checking.** One hallucinated fact can't hide inside an otherwise-correct answer.
- **Fail-safe.** If the LLM-judge returns something unparseable, the claim is treated as
  unverifiable with confidence 0, never silently passed.

## Installation

Requires Python 3.10+.

```bash
git clone https://github.com/VardanSingh2709/hallufirewall.git
cd hallufirewall
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[dev]"
```

`pip install -e .` installs only the core dependencies (`transformers`, `torch`). The `[dev]`
extra adds what the tests and examples need (`pytest`, `python-dotenv`, `google-genai`,
`sentence-transformers`).

The first run downloads the NLI model (`roberta-large-mnli`, about 1.4 GB). The RAG demo also
downloads a small embedding model (`all-MiniLM-L6-v2`, about 80 MB). Both are cached afterwards.

## Usage

```python
from hallufirewall import Firewall

def my_llm_client(prompt: str) -> str:
    # Call any LLM provider here and return its raw text response.
    ...

fw = Firewall(llm_client=my_llm_client)   # loads the NLI model once

result = fw.check(
    answer="Paris is the capital of France. It has a population of 50 million.",
    context=[
        "France is a country in Europe. Its capital city is Paris.",
        "Paris has a population of just over 2 million people.",
    ],
)

print(result.verdict)               # "block": the population claim is contradicted
print(result.overall_confidence)
for cv in result.claim_verdicts:
    print(cv.claim.text, "->", cv.label.value, round(cv.confidence, 2))
```

### Example `llm_client` using Gemini

Put your key in a `.env` file (copy `.env.example`; `.env` is gitignored):

```
GEMINI_API_KEY=your-key-here
```

```python
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

def gemini_llm_client(prompt: str) -> str:
    response = client.models.generate_content(model="gemini-flash-lite-latest", contents=prompt)
    return response.text
```

Model names on the Gemini free tier change often. `gemini-flash-lite-latest` is an alias that
follows the current Flash-Lite model, so it is less likely to break than a pinned version.

### Options

```python
Firewall(
    llm_client=my_llm_client,
    claim_extraction_mode="llm",       # "llm" (default, more accurate) or "rule_based" (free, naive)
    confidence_threshold=0.85,         # NLI confidence below this escalates to the LLM-judge
    nli_model_name="roberta-large-mnli",
)
```

## Tests

```bash
python -m pytest tests/
```

15 tests covering the decision policy, claim extraction (including a regression test for LLMs
that wrap JSON in markdown fences), and the end-to-end `Firewall` wiring. The wiring test uses
mocks, so the suite needs no model downloads or API keys.

## Examples

Run these from the repository root. All except `compare_nli_approaches.py` need `GEMINI_API_KEY`.

| Script | What it does |
|---|---|
| `examples/simple_rag_demo.py` | A small end-to-end RAG system (embedding retriever, Gemini generator) with the firewall on top, plus an interactive question loop |
| `examples/test_full_firewall.py` | Full pipeline on a multi-claim answer containing one hallucinated fact |
| `examples/test_hybrid.py` | Hybrid verifier on one true and one false claim |
| `examples/test_llm_judge_gemini.py` | LLM-judge alone |
| `examples/test_hard_cases.py` | Direct NLI vs LLM-judge on multi-hop, implicit negation, paraphrase and long-context cases |
| `examples/compare_nli_approaches.py` | Zero-shot NLI vs direct MNLI on easy and hard claims (no API key needed) |
| `examples/evaluate_ragtruth.py` | The RAGTruth evaluation described below |
| `examples/list_models.py` | Lists the Gemini models your API key can access |

```bash
python examples/simple_rag_demo.py
```

## Evaluation

Evaluated on `data/ragtruth_qa_eval_sample.json`: 40 examples from the QA test split of
[RAGTruth](https://github.com/ParticleMedia/RAGTruth), balanced at 20 human-labelled hallucinated
and 20 clean (sampled with `random.seed(42)`). An answer counts as "flagged" if the verdict is
anything other than `pass`.

|  | Flagged | Passed |
|---|---|---|
| **Actually hallucinated** | 18 (TP) | 2 (FN) |
| **Actually clean** | 10 (FP) | 10 (TN) |

| Metric | Value |
|---|---|
| Recall | 90.0% |
| Precision | 64.3% |
| F1 | 75.0% |

Recall is high by design, because the system leans toward flagging rather than risking a miss.
Precision pays for that: 10 of 20 clean answers were flagged unnecessarily.

```bash
python examples/evaluate_ragtruth.py
```

On the Gemini free tier this takes roughly 10 to 15 minutes, because the script deliberately
paces its calls (about 13 seconds apart) and retries with backoff when rate-limited. It writes
per-example results to `data/eval_results_log.json`, which is gitignored.

### Findings from development

- **Zero-shot NLI was unreliable; direct MNLI was not.** Using HuggingFace's
  `zero-shot-classification` pipeline for NLI got 3 of 5 test claims wrong (for example it
  marked "Paris is the capital of Germany" as supported). Feeding premise/hypothesis pairs
  directly to `roberta-large-mnli` got all 5 right with confidence of at least 0.96.
  `verifiers/nli.py` is kept as that documented baseline; `Firewall` uses `nli_direct.py`.
- **Direct NLI still fails on multi-hop claims.** A claim that needs facts from two separate
  chunks was labelled contradicted at only 0.54 confidence. That is below the 0.85 threshold,
  so the hybrid system escalates it, and the LLM-judge gets it right. This is the case for
  keeping both verifiers.

## Limitations

- The evaluation uses only 40 examples, so the numbers are indicative, not statistically robust.
- The 0.85 escalation threshold was chosen conservatively, not tuned against data.
- Two false negatives from the RAGTruth run have not been analysed individually.
- The LLM-judge's confidence is self-reported by the model and is less well calibrated than
  NLI's softmax probabilities.
- Rule-based claim extraction splits on punctuation only, so it fails on compound sentences
  and abbreviations like "Dr." (`"llm"` mode, the default, handles both).
- Free-tier LLM APIs have tight rate limits, which makes large evaluation runs slow.

## Project structure

```
hallufirewall/
    __init__.py
    types.py                 # Claim, ClaimVerdict, FirewallResult, Label
    claim_extraction.py      # rule-based and LLM-based claim extraction
    policy.py                # pass / flag / abstain / block decision logic
    firewall.py              # the public Firewall class
    verifiers/
        nli_direct.py        # NLI verifier (direct MNLI classification), used in production
        nli.py               # zero-shot NLI verifier, kept as a comparison baseline
        llm_judge.py         # LLM-as-judge verifier
        hybrid.py            # NLI first, escalate to LLM-judge when unsure
tests/                       # pytest suite
examples/                    # demo and evaluation scripts
data/                        # RAGTruth QA evaluation sample
```

## Acknowledgements

Evaluation data comes from RAGTruth ("RAGTruth: A Hallucination Corpus for Developing
Trustworthy Retrieval-Augmented Language Models").