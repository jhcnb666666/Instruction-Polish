"""Complexity heuristics to decide whether an instruction requires semantic LLM parsing."""

import re
from typing import Set


# Keywords that indicate temporal or logical structure beyond simple action sequences
TEMPORAL_KEYWORDS: Set[str] = {
    "before", "after", "until", "once", "when",
}

REPLACEMENT_NEGATION_KEYWORDS: Set[str] = {
    "instead of", "rather than", "do not", "without", "never",
}

# Phrasal verbs that the rule parser commonly misclassifies
PHRASAL_VERB_TRIGGERS = [
    r"\bfollow\b",
    r"\bcontinue\s+(straight|forward|on)\b",
    r"\b(take|make)\s+a\s+(left|right)\b",
]

# Structural markers
STRUCTURAL_SEMICOLON = ";"

# Action verbs used to count clauses
ACTION_VERBS = {
    "walk", "go", "move", "turn", "stop", "enter", "exit",
    "pass", "face", "wait", "follow", "continue", "take",
    "head", "proceed",
}

# Active vertical-movement patterns ( conservative — only flag ACTIVE motion )
ACTIVE_VERTICAL_PATTERNS = [
    r"\bgo\s+(up|down)\b.*\b(stairs|steps)\b",
    r"\b(take|ride|use|get)\b.*\b(elevator|lift)\b",
    r"\b(elevator|lift)\b.*\b(up|down)\b",
    r"\bflight\b.*\bof\b.*\bstairs\b",
    r"\b(first|second|third)\s+floor\b",
    r"\bupstairs\b",
    r"\bdownstairs\b",
    r"\bclimb\b",
    r"\bdescend\b",
    r"\bmove\s+to\s+the\s+\w+\s+floor\b",
]


def has_active_vertical_motion(text: str) -> bool:
    """Return True if the instruction contains active vertical navigation."""
    lower = text.lower()
    for pat in ACTIVE_VERTICAL_PATTERNS:
        if re.search(pat, lower):
            return True
    return False


def count_navigation_phases(text: str) -> int:
    """Count recognizable navigation action stages using the shared heuristic."""
    words = re.findall(r"\b\w+\b", text.lower())
    return sum(1 for word in words if word in ACTION_VERBS)


def requires_semantic_parser(text: str) -> bool:
    """
    Return True if the instruction has structural or semantic complexity
    that makes rule-based parsing unreliable.

    Heuristics (conservative — false-positive just routes to LLM, which is safe):
    - Temporal keywords: before, after, until, once, when
    - Replacement/negation: instead of, rather than, do not, without, never
    - Phrasal verbs: follow, take/make a left/right, continue straight
    - Structural: semicolon, or 3+ action clauses
    - Active vertical-movement: go up/down stairs, take elevator, upstairs, downstairs, floor
    """
    lower = text.lower()

    # Temporal
    for kw in TEMPORAL_KEYWORDS:
        if kw in lower:
            return True

    # Replacement / negation
    for kw in REPLACEMENT_NEGATION_KEYWORDS:
        if kw in lower:
            return True

    # Phrasal verbs
    for pat in PHRASAL_VERB_TRIGGERS:
        if re.search(pat, lower):
            return True

    # Structural: semicolon
    if STRUCTURAL_SEMICOLON in text:
        return True

    # Structural: three or more action clauses
    if count_navigation_phases(text) >= 3:
        return True

    # Active vertical-movement only
    if has_active_vertical_motion(text):
        return True

    return False
