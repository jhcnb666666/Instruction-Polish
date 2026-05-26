"""Built-in polishing strategies."""

import re
from typing import Callable, Dict, List

Strategy = Callable[[str], str]

# Global registry
_STRATEGIES: Dict[str, Strategy] = {}


def register_strategy(name: str, strategy: Strategy) -> None:
    """Register a custom strategy."""
    _STRATEGIES[name] = strategy


def get_strategy(name: str) -> Strategy:
    """Retrieve a strategy by name."""
    if name not in _STRATEGIES:
        raise KeyError(f"Strategy '{name}' not found. Available: {list(_STRATEGIES.keys())}")
    return _STRATEGIES[name]


def list_registered_strategies() -> List[str]:
    """List all registered strategy names."""
    return list(_STRATEGIES.keys())


def _ensure_punctuation(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".?!":
        text += "."
    return text


def _default_strategy(instruction: str) -> str:
    """Capitalize first letter and ensure punctuation."""
    text = instruction.strip()
    if text:
        text = text[0].upper() + text[1:]
    return _ensure_punctuation(text)


def strategy_clarity(instruction: str) -> str:
    """Improve clarity by removing filler words, fixing informal abbreviations, and standardizing punctuation."""
    fillers = ["kind of", "sort of", "kinda", "basically", "actually", "literally", "maybe", "perhaps"]
    result = instruction
    for filler in fillers:
        result = re.sub(rf"\b{filler}\b", "", result, flags=re.IGNORECASE)

    # Fix common informal abbreviations
    replacements = {
        r"\bu\b": "you",
        r"\bur\b": "your",
        r"\bwanna\b": "want to",
        r"\bgonna\b": "going to",
        r"\bdunno\b": "do not know",
        r"\bgimme\b": "give me",
        r"\bimo\b": "in my opinion",
        r"\bimho\b": "in my humble opinion",
        r"\btw\b": "by the way",
        r"\bbtw\b": "by the way",
        r"\bfyi\b": "for your information",
    }
    for pattern, repl in replacements.items():
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)

    # Clean up leftover punctuation / spacing at start (preserve non-ASCII)
    result = re.sub(r"^[\s\-–—.,;:!?]+", "", result).strip()

    # Capitalize first letter
    if result:
        result = result[0].upper() + result[1:]

    result = re.sub(r"\s+", " ", result).strip()
    return _ensure_punctuation(result)


def strategy_conciseness(instruction: str) -> str:
    """Make the instruction shorter while preserving intent."""
    result = re.sub(r"\s+", " ", instruction).strip()
    result = re.sub(r"(?i)\b(in order to|due to the fact that|with regard to|at this point in time|for the purpose of)\b",
                    lambda m: {"in order to": "to", "due to the fact that": "because",
                               "with regard to": "regarding", "at this point in time": "now",
                               "for the purpose of": "for"}
                    .get(m.group(0).lower(), m.group(0)), result)
    # Capitalize first letter
    if result:
        result = result[0].upper() + result[1:]
    return _ensure_punctuation(result)


def strategy_detail(instruction: str) -> str:
    """Add structure and request for detail."""
    result = instruction.strip()
    if not result:
        return _ensure_punctuation(result)
    # Normalize capitalization first
    result = result[0].upper() + result[1:]
    if not result.lower().startswith("please"):
        # Use comma for better flow if result starts with a prepositional phrase
        if result.lower().startswith(("in ", "on ", "at ", "to ", "for ", "with ")):
            result = "Please, " + result[0].lower() + result[1:]
        else:
            result = "Please " + result[0].lower() + result[1:]
    if "explain" not in result.lower() and "describe" not in result.lower():
        result = result.rstrip(".?!") + ", and explain your reasoning in detail."
    return _ensure_punctuation(result)


def strategy_roleplay(instruction: str) -> str:
    """Wrap instruction in a role-play context."""
    result = instruction.strip()
    if not result:
        return _ensure_punctuation(result)
    if "act as" not in result.lower() and "you are" not in result.lower():
        result = f"Act as an expert assistant. {result[0].upper()}{result[1:]}"
    return _ensure_punctuation(result)


# Built-in strategy names for auto-selection (does not include user-registered custom strategies)
_BUILTIN_STRATEGY_NAMES = ["default", "clarity", "conciseness", "detail", "roleplay"]


def strategy_auto(instruction: str) -> str:
    """Automatically pick the best built-in strategy based on heuristic score."""
    from .scoring import score_instruction

    candidates = []
    for name in _BUILTIN_STRATEGY_NAMES:
        fn = _STRATEGIES.get(name)
        if fn is None or name == "auto":
            continue
        try:
            polished = fn(instruction)
            score = score_instruction(polished)["score"]
            candidates.append((score, polished))
        except Exception:
            continue
    if not candidates:
        return _default_strategy(instruction)
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


# Register built-in strategies
register_strategy("default", _default_strategy)
register_strategy("clarity", strategy_clarity)
register_strategy("conciseness", strategy_conciseness)
register_strategy("detail", strategy_detail)
register_strategy("roleplay", strategy_roleplay)
register_strategy("auto", strategy_auto)
