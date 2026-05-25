"""Basic smoke tests for the CLI module."""

import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "instruction_polish", *args],
        capture_output=True,
        text=True,
    )


def test_list_strategies():
    result = run_cli("--list-strategies")
    assert result.returncode == 0
    assert "clarity" in result.stdout


def test_compare():
    result = run_cli("hello world", "--compare")
    assert result.returncode == 0
    assert "default" in result.stdout
    assert "clarity" in result.stdout


def test_analyze():
    result = run_cli("hello world", "--analyze")
    assert result.returncode == 0
    assert "score" in result.stdout


def test_diff():
    result = run_cli("hello world", "--diff")
    assert result.returncode == 0
    assert "--- original" in result.stdout
    assert "+++ polished" in result.stdout


def test_best():
    result = run_cli("u summarize this", "--best")
    assert result.returncode == 0
    assert "strategy:" in result.stdout
    assert "you" in result.stdout.lower()
