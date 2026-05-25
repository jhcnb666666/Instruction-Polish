"""Core logic for instruction polishing."""

from typing import Dict, Any

from .strategies import list_registered_strategies, get_strategy
from .scoring import score_instruction
from .utils import setup_logging

logger = setup_logging()


def list_strategies() -> list[str]:
    """Return available strategy names."""
    return list_registered_strategies()


def analyze(instruction: str) -> Dict[str, Any]:
    """
    Analyze an instruction and return quality metrics.

    Args:
        instruction: The raw instruction string.

    Returns:
        A dictionary with analysis results.
    """
    words = instruction.split()
    scoring = score_instruction(instruction)
    return {
        "length": len(instruction),
        "word_count": len(words),
        "has_question_mark": "?" in instruction,
        "avg_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0.0,
        "score": scoring["score"],
        "score_reasons": scoring["reasons"],
    }


def polish(instruction: str, strategy: str = "default") -> str:
    """
    Polish an instruction to improve clarity and effectiveness.

    Args:
        instruction: The raw instruction string.
        strategy: The polishing strategy to use. Can be a comma-separated list
                  to apply multiple strategies in sequence (e.g., "clarity,detail").

    Returns:
        The polished instruction.
    """
    result = instruction
    strategies = [s.strip() for s in strategy.split(",")]

    for strat in strategies:
        try:
            fn = get_strategy(strat)
        except KeyError:
            logger.warning("Unknown strategy '%s', falling back to 'default'", strat)
            fn = get_strategy("default")
        result = fn(result)
        logger.debug("Applied strategy '%s': %r", strat, result)

    return result
