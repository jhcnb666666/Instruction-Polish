"""Tests for heuristic scoring."""

from instruction_polish.scoring import score_instruction


def test_score_range():
    result = score_instruction("Explain quantum computing in detail.")
    assert 0 <= result["score"] <= 100


def test_short_instruction_penalty():
    low = score_instruction("hi")
    high = score_instruction("Explain quantum computing step by step with examples.")
    assert low["score"] < high["score"]


def test_informal_penalty():
    formal = score_instruction("Please explain your reasoning.")
    informal = score_instruction("u wanna explain this stuff somehow")
    assert informal["score"] < formal["score"]
    assert any("informal" in r for r in informal["reasons"])


def test_action_verb_bonus():
    with_verb = score_instruction("List three benefits of exercise.")
    without = score_instruction("Exercise is good.")
    assert with_verb["score"] > without["score"]
