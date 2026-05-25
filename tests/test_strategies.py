"""Tests for polishing strategies."""

from instruction_polish.strategies import (
    strategy_clarity,
    strategy_conciseness,
    strategy_detail,
    strategy_roleplay,
)


def test_clarity_removes_fillers_and_fixes_abbreviations():
    raw = "i kinda want u to sort of explain this."
    out = strategy_clarity(raw)
    assert "kind of" not in out.lower()
    assert "sort of" not in out.lower()
    assert " u " not in out.lower()
    assert out.startswith("I")
    assert out.endswith(".")


def test_conciseness_replaces_phrases():
    raw = "In order to win, you need to try hard."
    out = strategy_conciseness(raw)
    assert "in order to" not in out.lower()
    assert "to win" in out.lower()


def test_detail_adds_please():
    raw = "summarize quantum computing"
    out = strategy_detail(raw)
    assert out.lower().startswith("please")
    assert "explain your reasoning" in out.lower()


def test_roleplay_adds_prefix():
    raw = "help me write a poem"
    out = strategy_roleplay(raw)
    assert "act as" in out.lower()


def test_auto_returns_string():
    from instruction_polish.strategies import strategy_auto
    out = strategy_auto("hello world")
    assert isinstance(out, str)
    assert len(out) > 0
