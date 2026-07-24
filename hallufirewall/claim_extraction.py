"""
Claim extraction: turning a raw LLM answer into a list of Claim objects.

We start with a simple, rule-based approach (splitting on sentences).
Later in this file we'll add a smarter LLM-based approach, and both will
share the same "shape" (a function that takes a string, returns list[Claim])
so the rest of the system doesn't care which one is used underneath.
"""

import re
import json
from typing import Callable
from .types import Claim


def strip_markdown_json_fence(text: str) -> str:
    """
    Some LLMs wrap JSON responses in markdown code fences (```json ... ```)
    even when explicitly told not to. This strips that wrapping before
    we attempt to parse the response as JSON, so we don't silently treat
    a perfectly good response as unparseable.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    return cleaned


def split_into_sentences(text: str) -> list[str]:
    """
    A simple rule-based sentence splitter.

    NOTE: this is intentionally basic. It splits on '.', '!', '?' followed
    by a space. It will get confused by abbreviations like "Dr. Smith" or
    "e.g. this" -- that's a KNOWN limitation we'll discuss below.
    """
    text = text.strip()
    if not text:
        return []

    # This regex says: split the text at any '.', '!' or '?'
    # that is immediately followed by whitespace.
    # The parentheses in the pattern keep the punctuation attached
    # to the sentence it belongs to.
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Clean up: remove empty strings that can result from splitting
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def extract_claims_rule_based(answer: str) -> list[Claim]:
    """
    Rule-based claim extraction: one Claim per sentence.

    This is our v1 (baseline) extractor. It's fast and free, but treats
    each SENTENCE as one claim -- even if a sentence secretly contains
    two separate facts (e.g. "Paris is the capital of France and has
    2 million people" won't get split into two claims here).
    """
    sentences = split_into_sentences(answer)
    claims = [Claim(text=sentence, source_answer_span=sentence) for sentence in sentences]
    return claims


# This is the prompt template we'll send to the LLM. Notice it explicitly
# demands JSON output ONLY -- this makes the response easy to parse
# reliably, instead of trying to regex out claims from free-flowing prose.
CLAIM_EXTRACTION_PROMPT = """You are decomposing a piece of text into atomic factual claims.

Split the following ANSWER into a list of standalone factual claims. Each claim
must be understandable on its own, without needing the rest of the answer for context.
Do NOT include opinions, greetings, or filler -- only checkable facts.

Respond with ONLY a JSON array of strings, nothing else. No markdown, no explanation.

ANSWER:
{answer}

JSON array of claims:"""


def extract_claims_llm(answer: str, llm_client: Callable[[str], str]) -> list[Claim]:
    """
    LLM-based claim extraction.

    llm_client: a function that takes a prompt (str) and returns the model's
    raw text response (str). This is "dependency injection" -- we don't care
    HOW llm_client talks to whichever LLM provider, we just need something
    that behaves like: response_text = llm_client(prompt_text)
    """
    prompt = CLAIM_EXTRACTION_PROMPT.format(answer=answer)
    raw_response = llm_client(prompt)
    cleaned_response = strip_markdown_json_fence(raw_response)

    try:
        claim_texts = json.loads(cleaned_response)
    except json.JSONDecodeError:
        # If the LLM didn't return clean JSON (it happens sometimes),
        # we fall back to treating the whole answer as one claim rather
        # than crashing the whole pipeline.
        claim_texts = [answer]

    claims = [Claim(text=t, source_answer_span=t) for t in claim_texts]
    return claims
