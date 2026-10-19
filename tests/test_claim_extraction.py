"""
Formal tests for claim extraction.
"""

from hallufirewall.claim_extraction import (
    extract_claims_rule_based,
    split_into_sentences,
    strip_markdown_json_fence,
    extract_claims_llm,
)


def test_splits_multiple_sentences():
    text = "Paris is the capital of France. It has a population of 2 million."
    sentences = split_into_sentences(text)
    assert len(sentences) == 2


def test_empty_string_returns_no_claims():
    claims = extract_claims_rule_based("")
    assert claims == []


def test_single_sentence_returns_one_claim():
    claims = extract_claims_rule_based("Paris is the capital of France.")
    assert len(claims) == 1
    assert claims[0].text == "Paris is the capital of France."


def test_known_limitation_compound_sentence_stays_one_claim():
    text = "Paris is the capital of France and has a population of 2 million."
    claims = extract_claims_rule_based(text)
    assert len(claims) == 1


def test_strip_markdown_fence_removes_json_wrapper():
    wrapped = '```json\n["claim one", "claim two"]\n```'
    assert strip_markdown_json_fence(wrapped) == '["claim one", "claim two"]'


def test_strip_markdown_fence_leaves_plain_json_untouched():
    plain = '["claim one", "claim two"]'
    assert strip_markdown_json_fence(plain) == plain


def test_extract_claims_llm_handles_markdown_wrapped_response():
    def markdown_wrapping_client(prompt):
        return '```json\n["Paris is the capital of France", "It has 2 million people"]\n```'

    claims = extract_claims_llm("Paris is the capital of France. It has 2 million people.", markdown_wrapping_client)
    assert len(claims) == 2
    assert claims[0].text == "Paris is the capital of France"
