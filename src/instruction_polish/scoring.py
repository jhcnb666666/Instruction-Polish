"""Heuristic quality scoring for instructions."""

import re
from typing import Dict, Any


def score_instruction(instruction: str) -> Dict[str, Any]:
    """
    Compute a heuristic quality score (0-100) for an instruction.

    Higher is better. This is a rule-based approximation, not a model-based judge.
    """
    text = instruction.strip()
    lower = text.lower()
    words = text.split()
    word_count = len(words)

    score = 50.0  # start neutral
    reasons = []

    # Length penalties / bonuses
    if word_count < 3:
        score -= 20
        reasons.append("too short")
    elif word_count < 5:
        score -= 10
        reasons.append("very short")
    elif 8 <= word_count <= 40:
        score += 10
        reasons.append("good length")
    elif word_count > 80:
        score -= 10
        reasons.append("very long")

    # Action verbs encourage directness
    action_verbs = [
        "explain", "describe", "compare", "contrast", "list", "summarize",
        "write", "generate", "create", "analyze", "evaluate", "solve",
        "calculate", "translate", "convert", "find", "identify",
    ]
    has_action = any(v in lower for v in action_verbs)
    if has_action:
        score += 10
        reasons.append("contains action verb")
    else:
        score -= 5
        reasons.append("missing action verb")

    # Specificity markers
    specificity = ["step by step", "in detail", "with examples", "for beginners", "compared to"]
    if any(s in lower for s in specificity):
        score += 10
        reasons.append("has specificity markers")

    # Vague / weak words
    vague_words = ["stuff", "things", "something", "anything", "somehow", "maybe", "perhaps"]
    vague_count = sum(lower.count(w) for w in vague_words)
    if vague_count:
        score -= vague_count * 5
        reasons.append(f"contains {vague_count} vague word(s)")

    # Punctuation / formatting
    if text.endswith("?"):
        score += 5
        reasons.append("well-formed question")
    elif text.endswith(".") or text.endswith("!"):
        score += 2
        reasons.append("proper ending punctuation")
    else:
        score -= 5
        reasons.append("missing ending punctuation")

    # Capitalization
    if text and text[0].isupper():
        score += 3
        reasons.append("starts with capital letter")
    else:
        score -= 3
        reasons.append("missing capital letter")

    # Informal abbreviations
    informal = ["u ", "ur ", "wanna", "gonna", "dunno", "gimme", "kinda"]
    informal_count = sum(1 for w in informal if w in lower)
    if informal_count:
        score -= informal_count * 5
        reasons.append(f"contains {informal_count} informal abbreviation(s)")

    score = max(0.0, min(100.0, round(score, 1)))
    return {
        "score": score,
        "word_count": word_count,
        "reasons": reasons,
    }
