"""Vote aggregation logic for LLM-based VLN parsing — compact plan aggregation."""

import json
from typing import Any, Dict, List, Optional, Tuple

from .semantic_compiler import compile_draft, VALID_ACTIONS, VALID_DIRECTIONS


def normalize_landmark(landmark: str) -> str:
    """Normalize landmark for clustering comparison."""
    return " ".join(landmark.lower().split())


def _task_signature(task: Dict[str, Any]) -> Tuple[Any, ...]:
    """Create a task signature for clustering, normalizing feature order."""
    features = task.get("features", [])
    # Sort features by their content to make signature order-independent
    feature_sigs = tuple(
        sorted(
            (
                f.get("role"),
                f.get("relation"),
                normalize_landmark(f.get("landmark", "")),
                f.get("trigger"),
            )
            for f in features
        )
    )
    return (
        task.get("action", "UNKNOWN"),
        task.get("direction"),
        feature_sigs,
    )


def _plan_signature(plan: Dict[str, Any]) -> Tuple[Any, ...]:
    """Create a full plan signature for clustering."""
    tasks = plan.get("tasks", [])
    task_sigs = tuple(_task_signature(t) for t in tasks)
    constraints = tuple(sorted(json.dumps(c, sort_keys=True) for c in plan.get("constraints", [])))
    return (task_sigs, constraints)


def _is_valid_executable_plan(plan: Dict[str, Any]) -> bool:
    """Check if a compiled plan is a valid executable candidate."""
    if plan.get("status") != "ok":
        return False
    tasks = plan.get("tasks", [])
    if not tasks:
        return False
    for t in tasks:
        action = t.get("action", "UNKNOWN")
        if action == "UNKNOWN" or action not in VALID_ACTIONS:
            return False
        direction = t.get("direction")
        if direction is not None and direction not in VALID_DIRECTIONS:
            return False
    return True


def aggregate_votes(
    raw_votes: List[Dict[str, Any]],
    original_instruction: str,
) -> List[Dict[str, Any]]:
    """
    Aggregate multiple LLM semantic-draft outputs into ranked compact candidate plans.

    Strategy:
    1. Compile each draft into a compact plan.
    2. Group valid executable plans by their full plan signature.
    3. Keep top-3 different compact plans, each with candidate_id and vote_support.

    Returns:
        List of compact plan dicts, each with extra keys:
        - "candidate_id": "p1", "p2", or "p3"
        - "vote_support": int (number of raw votes matching this plan)
        Sorted by vote_support descending.
    """
    if not raw_votes:
        return []

    # Compile drafts to compact plans
    compiled_plans: List[Dict[str, Any]] = []
    invalid_votes_seen = False
    for vote in raw_votes:
        plan = compile_draft(vote)
        if _is_valid_executable_plan(plan):
            compiled_plans.append(plan)
        else:
            invalid_votes_seen = True

    if not compiled_plans:
        return []

    # Group by plan signature
    plan_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for plan in compiled_plans:
        sig = _plan_signature(plan)
        plan_groups.setdefault(sig, []).append(plan)

    # Sort groups by size descending
    sorted_groups = sorted(plan_groups.items(), key=lambda item: len(item[1]), reverse=True)

    # Build ranked candidate plans (max 3)
    ranked: List[Dict[str, Any]] = []
    for idx, (sig, group) in enumerate(sorted_groups[:3], start=1):
        representative = group[0]
        candidate: Dict[str, Any] = {
            "candidate_id": f"p{idx}",
            "vote_support": len(group),
            "status": representative.get("status", "ok"),
            "tasks": representative.get("tasks", []),
            "constraints": _union_dicts([p.get("constraints", []) for p in group]),
        }
        if representative.get("reason") is not None:
            candidate["reason"] = representative["reason"]
        # Flag if any invalid votes were filtered in this round
        if invalid_votes_seen:
            candidate["_invalid_votes_seen"] = True
        ranked.append(candidate)

    return ranked


def _union_dicts(dict_lists: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Union unique dicts across multiple lists (by JSON serialization)."""
    seen: set = set()
    result = []
    for lst in dict_lists:
        for d in lst:
            key = json.dumps(d, sort_keys=True)
            if key not in seen:
                seen.add(key)
                result.append(d)
    return result


def get_conflict_candidates(
    raw_votes: List[Dict[str, Any]]
) -> List[List[Dict[str, Any]]]:
    """
    Extract unique conflicting full plans from votes for adjudication.

    Returns:
        List of unique candidate plans, each a list of step dicts
        with keys action, direction, features.
    """
    seen: set = set()
    candidates = []
    for vote in raw_votes:
        plan = compile_draft(vote)
        tasks = plan.get("tasks", [])
        step_plan = [
            {
                "action": t.get("action", "UNKNOWN"),
                "direction": t.get("direction"),
                "features": t.get("features", []),
            }
            for t in tasks
        ]
        plan_tuple = json.dumps(step_plan, sort_keys=True)
        if plan_tuple not in seen:
            seen.add(plan_tuple)
            candidates.append(step_plan)
    return candidates
