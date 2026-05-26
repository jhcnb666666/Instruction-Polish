"""Vote aggregation logic for LLM-based VLN parsing — compact plan aggregation."""

import json
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .semantic_compiler import compile_draft


def normalize_landmark(landmark: str) -> str:
    """Normalize landmark for clustering comparison."""
    return " ".join(landmark.lower().split())


def _plan_signature(plan: Dict[str, Any]) -> Tuple[Any, ...]:
    """Create a full plan signature for clustering."""
    tasks = plan.get("tasks", [])
    task_sigs = tuple(
        (
            t.get("action", "UNKNOWN"),
            t.get("direction"),
            tuple(
                (f.get("role"), f.get("relation"), normalize_landmark(f.get("landmark", "")), f.get("trigger"))
                for f in t.get("features", [])
            ),
        )
        for t in tasks
    )
    constraints = tuple(sorted(json.dumps(c, sort_keys=True) for c in plan.get("constraints", [])))
    return (task_sigs, constraints)


def _action_sequence(plan: Dict[str, Any]) -> List[str]:
    """Extract just the action sequence from a plan."""
    return [t.get("action", "UNKNOWN") for t in plan.get("tasks", [])]


def aggregate_votes(
    raw_votes: List[Dict[str, Any]],
    original_instruction: str,
) -> List[Dict[str, Any]]:
    """
    Aggregate multiple LLM semantic-draft outputs into ranked compact candidate plans.

    Strategy:
    1. Compile each draft into a compact plan.
    2. Group plans by their full plan signature (tasks + constraints).
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
    for vote in raw_votes:
        plan = compile_draft(vote)
        # Skip unsupported plans from compilation; they don't form viable candidates
        if plan.get("status") == "unsupported":
            continue
        compiled_plans.append(plan)

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
