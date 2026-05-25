"""Tests for strategy suggestion."""

from instruction_polish.suggest import suggest_strategy


def test_suggests_clarity_for_informal():
    result = suggest_strategy("u wanna explain this")
    assert result["strategy"] == "clarity"


def test_suggests_conciseness_for_wordy():
    result = suggest_strategy("In order to win, try hard")
    assert result["strategy"] == "conciseness"


def test_suggests_default_for_good():
    result = suggest_strategy("Please explain quantum computing step by step with examples.")
    assert result["strategy"] in ("default", "detail")
