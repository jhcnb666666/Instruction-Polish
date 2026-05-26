"""Semantic compiler: convert internal semantic drafts to compact pruned plans."""

from typing import Any, Dict, List, Optional, Set, Tuple

VALID_FEATURE_ROLES = {
    "path",
    "where",
    "progress",
    "target",
    "terminate",
    "start",
}

VALID_ACTIONS = {
    "MOVE_FORWARD",
    "TURN",
    "GO_TO",
    "PASS",
    "ENTER",
    "EXIT",
    "STOP",
    "FACE",
    "WAIT",
    "UNKNOWN",
}

VALID_DIRECTIONS = {
    "left",
    "right",
    "forward",
    "backward",
    "straight",
    "around",
}

VALID_RELATIONS = {
    "near",
    "in_front_of",
    "behind",
    "left_of",
    "right_of",
    "left_of_agent",
    "right_of_agent",
    "inside",
    "into",
    "outside",
    "through",
    "along",
    "toward",
    "away_from",
    "at",
    "between",
    "end_of",
    "past",
    "before",
    "just_before",
    "before_hitting",
    "after",
}


def compile_draft(draft: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compile an internal semantic draft into a compact candidate plan.

    Steps:
    1. Validate action ids are unique.
    2. Validate feature roles/relations.
    3. Delete excluded actions.
    4. Convert safety-relevant negations to constraints.
    5. Topological sort by internal order.
    6. Assign continuous step_id.
    7. Build compact tasks and constraints.
    8. Detect unsupported vertical motion.
    """
    if not isinstance(draft, dict):
        return _invalid_draft("invalid_draft")

    actions = draft.get("actions", [])
    order = draft.get("order", [])
    excluded = draft.get("excluded", [])
    constraints_raw = draft.get("constraints", [])

    if not isinstance(actions, list):
        return _invalid_draft("invalid_draft")

    # Auto-assign ids if missing
    for i, a in enumerate(actions):
        if "id" not in a:
            a["id"] = f"a{i+1}"

    # 1. Validate unique ids
    ids: Set[str] = set()
    for a in actions:
        aid = a.get("id")
        if not aid or aid in ids:
            return _invalid_draft("duplicate_or_missing_action_id")
        ids.add(aid)

    # Validate actions and directions before anything else
    for a in actions:
        act = a.get("action", "")
        if act and act not in VALID_ACTIONS:
            return _invalid_draft(f"invalid_action:{act}")
        direction = a.get("direction")
        if direction is not None and direction not in VALID_DIRECTIONS:
            return _invalid_draft(f"invalid_direction:{direction}")

    # 2. Validate features — skip unknown roles/relations rather than failing entirely
    for a in actions:
        valid_features = []
        for f in a.get("features", []):
            role = f.get("role")
            if role not in VALID_FEATURE_ROLES:
                continue
            relation = f.get("relation")
            if relation is not None and relation not in VALID_RELATIONS:
                # Skip features with invalid relations
                continue
            if role in ("terminate", "start"):
                if not f.get("trigger") or not f.get("landmark"):
                    continue
            else:
                if not f.get("relation") or not f.get("landmark"):
                    continue
            valid_features.append(f)
        a["features"] = valid_features

    # 3. Prune excluded actions BEFORE vertical detection
    excluded_ids = set()
    for ex in excluded:
        if isinstance(ex, dict):
            excluded_ids.add(ex.get("id"))
        elif isinstance(ex, str):
            excluded_ids.add(ex)

    remaining_actions = [a for a in actions if a.get("id") not in excluded_ids]

    # 8. Detect vertical motion in remaining actions only
    for a in remaining_actions:
        if _is_active_vertical_action(a):
            return {
                "status": "unsupported",
                "confidence": 1.0,
                "tasks": [],
                "constraints": [],
                "reason": "vertical_motion_not_supported",
            }

    # 4. Convert safety constraints
    constraints: List[Dict[str, Any]] = []
    for cr in constraints_raw:
        if cr.get("type") == "forbidden_action":
            action = cr.get("action", "UNKNOWN")
            if action not in VALID_ACTIONS:
                return _invalid_draft(f"invalid_constraint_action:{action}")
            constraint: Dict[str, Any] = {
                "type": "forbidden_action",
                "action": action,
                "features": _normalize_features(cr.get("features", [])),
            }
            if cr.get("direction"):
                constraint["direction"] = cr["direction"]
            constraints.append(constraint)

    # 5. Topological sort
    topo_result = _topological_sort(remaining_actions, order)
    if isinstance(topo_result, dict):
        # Cycle detected
        return topo_result
    sorted_actions = topo_result

    # 6. Assign step_id and build compact tasks
    tasks: List[Dict[str, Any]] = []
    for step_id, a in enumerate(sorted_actions, start=1):
        task: Dict[str, Any] = {
            "step_id": step_id,
            "action": a.get("action", "UNKNOWN"),
            "features": _normalize_features(a.get("features", [])),
            "confidence": round(a.get("confidence", 1.0), 2),
        }
        direction = a.get("direction")
        if direction and direction in VALID_DIRECTIONS:
            task["direction"] = direction
        tasks.append(task)

    # Check for UNKNOWN actions in the final plan
    for t in tasks:
        if t.get("action") == "UNKNOWN":
            return _invalid_draft("unknown_action_in_plan")

    # Build result
    result: Dict[str, Any] = {
        "status": "ok",
        "confidence": draft.get("confidence", 1.0),
        "tasks": tasks,
        "constraints": constraints,
    }
    return result


def _invalid_draft(reason: str) -> Dict[str, Any]:
    return {
        "status": "needs_review",
        "confidence": 0.0,
        "tasks": [],
        "constraints": [],
        "reason": reason,
    }


def _unsupported(reason: str) -> Dict[str, Any]:
    return {
        "status": "unsupported",
        "confidence": 1.0,
        "tasks": [],
        "constraints": [],
        "reason": reason,
    }


def _normalize_features(features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize and deduplicate features, removing empty optional fields."""
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for f in features:
        nf: Dict[str, Any] = {"role": f["role"]}
        if f.get("relation"):
            nf["relation"] = f["relation"]
        if f.get("landmark"):
            nf["landmark"] = f["landmark"]
        if f.get("trigger"):
            nf["trigger"] = f["trigger"]
        key = "|".join(f"{k}={v}" for k, v in sorted(nf.items()))
        if key not in seen:
            seen.add(key)
            out.append(nf)
    return out


def _topological_sort(actions: List[Dict[str, Any]], order: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Topologically sort actions based on order constraints."""
    # Build adjacency list: id -> list of ids that must come after
    graph: Dict[str, Set[str]] = {a["id"]: set() for a in actions}
    in_degree: Dict[str, int] = {a["id"]: 0 for a in actions}

    for o in order:
        before = o.get("before")
        after = o.get("after")
        if before in graph and after in graph and after not in graph[before]:
            graph[before].add(after)
            in_degree[after] += 1

    # Kahn's algorithm
    queue = [aid for aid, deg in in_degree.items() if deg == 0]
    # Deterministic ordering: sort queue by original action index
    action_index = {a["id"]: i for i, a in enumerate(actions)}
    queue.sort(key=lambda aid: action_index[aid])

    result: List[Dict[str, Any]] = []
    while queue:
        queue.sort(key=lambda aid: action_index[aid])
        aid = queue.pop(0)
        action = next(a for a in actions if a["id"] == aid)
        result.append(action)
        for neighbor in sorted(graph[aid], key=lambda x: action_index[x]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(result) != len(actions):
        # Cycle detected; return invalid_draft
        return _invalid_draft("cyclic_execution_order")

    return result


def _is_active_vertical_action(action: Dict[str, Any]) -> bool:
    """Check if an action describes active vertical movement."""
    act = action.get("action", "")
    features = action.get("features", [])
    direction = action.get("direction", "")

    # Collect landmarks from features
    landmarks = [f.get("landmark", "").lower() for f in features if f.get("landmark")]
    landmark_str = " ".join(landmarks)

    text = f"{act} {direction} {landmark_str}".lower()

    active_vertical_patterns = [
        r"\bgo\s+(up|down)\b.*\b(stairs|steps)\b",
        r"\b(stairs|steps)\b.*\b(up|down)\b",
        r"\b(take|ride|use|get)\b.*\b(elevator|lift)\b",
        r"\b(elevator|lift)\b.*\b(up|down)\b",
        r"\bclimb\b",
        r"\bdescend\b",
        r"\bupstairs\b",
        r"\bdownstairs\b",
        r"\b(first|second|third)\s+floor\b",
        r"\bmove\s+to\s+the\s+\w+\s+floor\b",
    ]

    import re
    for pat in active_vertical_patterns:
        if re.search(pat, text):
            return True

    return False
