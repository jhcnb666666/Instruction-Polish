"""Top-3 candidate extraction for each instruction segment — compact feature output."""

import copy
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
    Extract top-3 candidate parses for a single segment.

    Each candidate is a compact dict with:
    - action
    - direction (optional)
    - features: list of compact feature dicts
    - confidence
    """
    base = _extract_base_candidate(segment)

    # If 3D phrase detected, force low confidence and UNKNOWN action
    if has_3d_phrase(segment):
        base["action"] = "UNKNOWN"
        base["direction"] = None
        base["confidence"] = max(0.0, base["confidence"] - 0.30)
        cands = [copy.deepcopy(base) for _ in range(3)]
        cands[0]["confidence"] = max(0.0, cands[0]["confidence"])
        cands[1]["confidence"] = max(0.0, cands[1]["confidence"] - 0.05)
        cands[2]["confidence"] = max(0.0, cands[2]["confidence"] - 0.10)
        return sort_candidates(cands)

    candidates = [base]

    # Candidate 2: alternate direction if direction was found
    alt2 = copy.deepcopy(base)
    if base.get("direction") is not None:
        alt2["direction"] = _alternate_direction(base["direction"])
        alt2["confidence"] = max(0.0, alt2["confidence"] - 0.08)
    else:
        # perturb relation in features instead
        alt2["features"] = _perturb_relation_in_features(alt2["features"])
        alt2["confidence"] = max(0.0, alt2["confidence"] - 0.05)
    candidates.append(alt2)

    # Candidate 3: alternate action or perturb
    alt3 = copy.deepcopy(base)
    if base["action"] in ("GO_TO", "MOVE_FORWARD"):
        alt3["action"] = "GO_TO" if base["action"] == "MOVE_FORWARD" else "MOVE_FORWARD"
        alt3["confidence"] = max(0.0, alt3["confidence"] - 0.10)
    elif base["action"] == "UNKNOWN":
        alt3["confidence"] = max(0.0, alt3["confidence"] - 0.03)
    else:
        alt3["features"] = _perturb_relation_in_features(alt3["features"])
        alt3["confidence"] = max(0.0, alt3["confidence"] - 0.06)
    candidates.append(alt3)

    return sort_candidates(candidates)


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


def _alternate_direction(direction: str) -> str:
    """Return a plausible alternative direction."""
    opposites = {
        "left": "right",
        "right": "left",
        "forward": "backward",
        "backward": "forward",
        "straight": "forward",
        "around": "left",
    }
    return opposites.get(direction, "left")


def _perturb_relation_in_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Toggle relation between 'at' and 'near' in features."""
    out = []
    for f in features:
        nf = dict(f)
        if nf.get("relation") == "at":
            nf["relation"] = "near"
        elif nf.get("relation") == "near":
            nf["relation"] = "at"
        out.append(nf)
    return out


def sort_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort candidates by confidence descending."""
    return sorted(candidates, key=lambda c: c["confidence"], reverse=True)
