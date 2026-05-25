"""Normalization of actions, directions, and spatial relations."""

import re
from typing import Tuple

# Action mappings: verb phrase -> ACTION label
ACTION_MAP = {
    "walk": "MOVE_FORWARD",
    "go": "GO_TO",
    "move": "MOVE_FORWARD",
    "turn": "TURN",
    "stop": "STOP",
    "enter": "ENTER",
    "exit": "EXIT",
    "pass": "PASS",
    "face": "FACE",
    "wait": "WAIT",
    "follow": "MOVE_FORWARD",
    "continue": "MOVE_FORWARD",
    "head": "MOVE_FORWARD",
}

# Direction mappings
DIRECTION_MAP = {
    "left": "left",
    "right": "right",
    "forward": "forward",
    "forwards": "forward",
    "backward": "backward",
    "backwards": "backward",
    "back": "backward",
    "straight": "straight",
    "around": "around",
}

# Spatial relation phrase -> relation label
RELATION_PATTERNS = [
    (r"\bin front of\b", "in_front_of"),
    (r"\bleft of\b", "left_of"),
    (r"\bright of\b", "right_of"),
    (r"\bin the middle of\b", "between"),
    (r"\bbetween\b", "between"),
    (r"\bnear\b", "near"),
    (r"\bnext to\b", "near"),
    (r"\bbeside\b", "near"),
    (r"\bbehind\b", "behind"),
    (r"\bthrough\b", "through"),
    (r"\balong\b", "along"),
    (r"\btoward\b", "toward"),
    (r"\btowards\b", "toward"),
    (r"\baway from\b", "away_from"),
    (r"\bat\b", "at"),
    (r"\binto\b", "inside"),
    (r"\binside\b", "inside"),
    (r"\boutside\b", "outside"),
    (r"\bpast\b", "past"),
    (r"\bend of\b", "end_of"),
    (r"\bby\b", "near"),
    (r"\bdown\b", "along"),
    (r"\bup\b", "along"),
    (r"\bto\b", "toward"),
]

# 3D / unsupported keywords (pure vertical movement)
UNSUPPORTED_3D_KEYWORDS = {
    "upstairs", "downstairs",
    "climb", "descend",
}

# Elevator-related: only flag as 3D when the verb implies taking/riding it
ELEVATOR_3D_VERBS = {"take", "ride", "use", "get"}

# Common VLN action patterns (for confidence boost)
COMMON_VLN_PATTERNS = [
    r"\bwalk\b.*\bto\b",
    r"\bturn\b\s+(left|right)\b",
    r"\bgo\b\s+(straight|forward|backward|back)\b",
    r"\bstop\b",
    r"\benter\b",
    r"\bexit\b",
    r"\bface\b",
]


def normalize_action(text: str) -> Tuple[str, float]:
    """
    Detect the primary action verb and map to a canonical action label.
    Returns (action_label, confidence_contribution).
    """
    lower = text.lower()
    words = lower.split()
    if not words:
        return "UNKNOWN", 0.0

    first_word = words[0]

    # Check for unsupported 3D action first
    for kw in UNSUPPORTED_3D_KEYWORDS:
        if kw in words:
            return "UNKNOWN", 0.0

    # Special handling for "go": GO_TO if "to" is present, else MOVE_FORWARD
    if first_word == "go":
        if "to" in words:
            return "GO_TO", 0.35
        return "MOVE_FORWARD", 0.35

    if first_word in ACTION_MAP:
        return ACTION_MAP[first_word], 0.35

    # Check if any action word appears early in the sentence
    for i, w in enumerate(words[:3]):
        if w in ACTION_MAP:
            if w == "go":
                return ("GO_TO" if "to" in words else "MOVE_FORWARD"), 0.30
            return ACTION_MAP[w], 0.30

    return "UNKNOWN", 0.0


def normalize_direction(text: str) -> Tuple[str, float]:
    """
    Extract direction from text.
    Returns (direction_label, confidence_contribution).
    """
    lower = text.lower()

    # Handle special case: "turn around" -> around
    if "turn around" in lower or "turning around" in lower:
        return "around", 0.10

    # Handle "go back" -> backward
    if re.search(r"\bgo\s+back\b", lower):
        return "backward", 0.10

    # Handle phrasal direction expressions
    if re.search(r"\b(take|make)\s+a\s+left\b", lower):
        return "left", 0.10
    if re.search(r"\b(take|make)\s+a\s+right\b", lower):
        return "right", 0.10

    for word, label in DIRECTION_MAP.items():
        if re.search(rf"\b{re.escape(word)}\b", lower):
            return label, 0.10

    return "unknown", 0.0


def normalize_relation(text: str) -> Tuple[str, float]:
    """
    Extract spatial relation from text.
    Returns (relation_label, confidence_contribution).
    """
    lower = text.lower()
    for pattern, label in RELATION_PATTERNS:
        if re.search(pattern, lower):
            return label, 0.15
    return "unknown", 0.0


def has_3d_phrase(text: str) -> bool:
    """Check if text contains ACTIVE unsupported 3D navigation phrases.

    Only flags phrases where the instruction is asking the robot to
    actively perform vertical movement (e.g., "go upstairs", "take the
    elevator up").  Passive references like "walk past the stairs" or
    "stop beside the elevator" are NOT flagged.
    """
    lower = text.lower()
    words = set(lower.split())

    # Explicit vertical adverbs
    if words & UNSUPPORTED_3D_KEYWORDS:
        return True

    # Elevator: only 3D when verb implies riding/taking it
    if "elevator" in words or "lift" in words:
        for verb in ELEVATOR_3D_VERBS:
            if re.search(rf"\b{verb}\b.*\b(elevator|lift)\b", lower):
                return True
            if re.search(rf"\b(elevator|lift)\b.*\bup\b", lower):
                return True
            if re.search(rf"\b(elevator|lift)\b.*\bdown\b", lower):
                return True
        return False

    # Floor references
    if re.search(r"\b(second|third|first)\s+floor\b", lower):
        return True
    if re.search(r"\bmove\s+to\s+the\s+\w+\s+floor\b", lower):
        return True

    # Active stair patterns
    if re.search(r"\bgo\s+up\s+(the\s+)?stairs\b", lower):
        return True
    if re.search(r"\bgo\s+down\s+(the\s+)?stairs\b", lower):
        return True
    if re.search(r"\bclimb\b", lower):
        return True
    if re.search(r"\bdescend\b", lower):
        return True

    return False


def is_common_vln_pattern(text: str) -> bool:
    """Check if text matches a common VLN pattern."""
    lower = text.lower()
    for pat in COMMON_VLN_PATTERNS:
        if re.search(pat, lower):
            return True
    return False


def extract_landmark(text: str, relation: str, action: str) -> Tuple[str, float]:
    """
    Heuristic extraction of landmark/location from segment text.
    Returns (landmark, confidence_contribution).
    """
    lower = text.lower()

    # Remove common leading verbs/adverbs for landmark extraction
    cleaned = lower
    for verb in ACTION_MAP:
        cleaned = re.sub(rf"^\b{verb}\b\s+", "", cleaned)

    # Remove direction words
    for d in DIRECTION_MAP:
        cleaned = re.sub(rf"\b{d}\b", "", cleaned)

    # Remove relation phrases to get the object
    for pattern, _ in RELATION_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned)

    # Remove temporal / filler phrases
    cleaned = re.sub(r"\bfor\s+a\s+(moment|while|second|bit)\b", "", cleaned)
    cleaned = re.sub(r"\bfor\s+(moment|moment|some\s+time)\b", "", cleaned)

    # Remove articles and extra words
    cleaned = re.sub(r"\b(the|a|an|to|and|then|it|there|here|over)\b", "", cleaned)
    cleaned = re.sub(r"[\s,]+", " ", cleaned).strip()

    if not cleaned:
        return "", 0.0

    # If the cleaned text is just a single word or short phrase, use it
    if len(cleaned.split()) <= 3:
        return cleaned, 0.25

    # Otherwise try to pick the last noun phrase (simplistic)
    tokens = cleaned.split()
    landmark = " ".join(tokens[-2:]) if len(tokens) > 1 else tokens[0]
    return landmark, 0.20
