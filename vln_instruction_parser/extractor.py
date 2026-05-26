"""Top candidate extraction for each instruction segment — compact feature output."""

from typing import List, Dict, Any

from .normalizer import (
    normalize_action,
    normalize_direction,
    normalize_relation,
    extract_landmark,
    has_3d_phrase,
    is_common_vln_pattern,
)


def extract_candidates(segment: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract the best candidate parse for a single segment.

    Returns a single-candidate list to maintain interface compatibility.
    Alternative candidates must come from actual LLM semantic interpretation,
    not from synthetic perturbation.
    """
    base = _extract_base_candidate(segment)

    # If 3D phrase detected, force low confidence and UNKNOWN action
    if has_3d_phrase(segment):
        base["action"] = "UNKNOWN"
        base["direction"] = None
        base["confidence"] = max(0.0, base["confidence"] - 0.30)
        return sort_candidates([base])

    return sort_candidates([base])


def _extract_base_candidate(segment: str) -> Dict[str, Any]:
    """Extract the base (best) compact candidate for a segment."""
    action, action_conf = normalize_action(segment)
    direction, dir_conf = normalize_direction(segment)
    relation, rel_conf = normalize_relation(segment)
    landmark, lm_conf = extract_landmark(segment, relation, action)

    confidence = action_conf + lm_conf + rel_conf + dir_conf
    if is_common_vln_pattern(segment):
        confidence += 0.10

    if action == "UNKNOWN":
        confidence = max(0.0, confidence - 0.20)

    confidence = min(1.0, max(0.0, confidence))

    candidate: Dict[str, Any] = {
        "action": action,
        "confidence": confidence,
        "features": [],
    }

    if direction and direction != "unknown":
        candidate["direction"] = direction

    if landmark and relation and relation != "unknown":
        candidate["features"].append({
            "role": "where",
            "relation": relation,
            "landmark": landmark,
        })
    elif landmark:
        candidate["features"].append({
            "role": "where",
            "relation": "at",
            "landmark": landmark,
        })

    return candidate


def sort_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort candidates by confidence descending."""
    return sorted(candidates, key=lambda c: c["confidence"], reverse=True)
