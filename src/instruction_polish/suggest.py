"""Strategy suggestion based on heuristic analysis."""

from typing import Dict, Any

from .scoring import score_instruction


def suggest_strategy(instruction: str) -> Dict[str, Any]:
    """
    Analyze an instruction and suggest the most appropriate strategy.

    Returns a dict with 'strategy', 'reason', and 'score'.
    """
    lower = instruction.lower()
    scoring = score_instruction(instruction)
    reasons = []

    # Rule-based suggestions
    informal = ["u ", "ur ", "wanna", "gonna", "dunno", "gimme", "kinda", "sorta"]
    has_informal = any(w in lower for w in informal)

    fillers = ["kind of", "sort of", "basically", "actually", "literally", "maybe", "perhaps"]
    has_fillers = any(f in lower for f in fillers)

    wordy = ["in order to", "due to the fact that", "with regard to", "at this point in time"]
    has_wordy = any(w in lower for w in wordy)

    if has_informal or has_fillers:
        reasons.append("contains informal language or filler words")
        return {
            "strategy": "clarity",
            "reason": "; ".join(reasons),
            "score": scoring["score"],
        }

    if has_wordy:
        reasons.append("contains wordy phrases")
        return {
            "strategy": "conciseness",
            "reason": "; ".join(reasons),
            "score": scoring["score"],
        }

    if scoring["score"] < 50:
        reasons.append("low heuristic score suggests need for improvement")
        return {
            "strategy": "auto",
            "reason": "; ".join(reasons),
            "score": scoring["score"],
        }

    if "?" in instruction and scoring["score"] >= 60:
        reasons.append("question with decent score; detail may help")
        return {
            "strategy": "detail",
            "reason": "; ".join(reasons),
            "score": scoring["score"],
        }

    reasons.append("instruction looks okay; default polish applied")
    return {
        "strategy": "default",
        "reason": "; ".join(reasons),
        "score": scoring["score"],
    }
