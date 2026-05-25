"""Unit tests for core module."""

from instruction_polish.core import analyze, polish, list_strategies


def test_list_strategies():
    strategies = list_strategies()
    assert "default" in strategies
    assert "clarity" in strategies


def test_analyze_basic():
    result = analyze("Hello world")
    assert result["length"] == 11
    assert result["word_count"] == 2
    assert result["has_question_mark"] is False
    assert result["avg_word_length"] == 5.0


def test_polish_adds_punctuation():
    assert polish("Hello world") == "Hello world."


def test_polish_keeps_existing_punctuation():
    assert polish("Hello world?") == "Hello world?"


def test_polish_unknown_strategy_fallback():
    assert polish("Hello world", strategy="nonexistent") == "Hello world."


def test_polish_multiple_strategies():
    # clarity fixes "u", detail adds "Please" and detail request
    result = polish("u summarize this", strategy="clarity,detail")
    assert "Please" in result
    assert "you" in result.lower()
    assert "explain your reasoning" in result.lower()
