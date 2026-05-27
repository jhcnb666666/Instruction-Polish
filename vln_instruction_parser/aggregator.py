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


def extract_valid_plan_pool(
    raw_votes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compile all raw votes and return every distinct valid executable plan.

    Unlike aggregate_votes which only returns top-3 signatures,
    this preserves all valid plans for downstream local step-candidate extraction.

    Returns:
        List of valid compact plan dicts (each with status="ok" and non-empty tasks).
    """
    if not raw_votes:
        return []

    seen: set = set()
    pool: List[Dict[str, Any]] = []
    for vote in raw_votes:
        plan = compile_draft(vote)
        if not _is_valid_executable_plan(plan):
            continue
        sig = _plan_signature(plan)
        if sig not in seen:
            seen.add(sig)
            pool.append(plan)
    return pool


def extract_step_candidates(
    primary_plan: Dict[str, Any],
    all_valid_plans: List[Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Compare all valid plans against the selected primary plan and extract
    local step-level candidates that differ in exactly one task.

    Rules:
    - Candidate must have same number of tasks as primary, in same order.
    - Constraints must be identical.
    - Exactly one task may differ.
    - The differing task maps to the corresponding primary step_id.

    Returns:
        Dict mapping step_id -> list of differing task dicts (candidate tasks).
    """
    primary_tasks = primary_plan.get("tasks", [])
    primary_constraints = primary_plan.get("constraints", [])
    primary_constraint_sig = tuple(sorted(json.dumps(c, sort_keys=True) for c in primary_constraints))

    if not primary_tasks:
        return {}

    candidates_by_step: Dict[int, List[Dict[str, Any]]] = {}

    for plan in all_valid_plans:
        # Skip the primary plan itself
        if _plan_signature(plan) == _plan_signature(primary_plan):
            continue

        tasks = plan.get("tasks", [])
        constraints = plan.get("constraints", [])

        # Same task count
        if len(tasks) != len(primary_tasks):
            continue

        # Same constraints
        constraint_sig = tuple(sorted(json.dumps(c, sort_keys=True) for c in constraints))
        if constraint_sig != primary_constraint_sig:
            continue

        # Find differing tasks
        diffs = []
        for i, (pt, ct) in enumerate(zip(primary_tasks, tasks)):
            if not _task_eq(pt, ct):
                diffs.append((i, ct))

        # Exactly one task differs
        if len(diffs) != 1:
            continue

        step_idx, candidate_task = diffs[0]
        step_id = primary_tasks[step_idx].get("step_id", step_idx + 1)
        candidates_by_step.setdefault(step_id, []).append(candidate_task)

    return candidates_by_step


def _task_eq(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Check whether two task dicts are identical for diff comparison."""
    if a.get("action") != b.get("action"):
        return False
    if a.get("direction") != b.get("direction"):
        return False
    # Compare features as sorted tuples for order-independent comparison
    a_features = a.get("features", [])
    b_features = b.get("features", [])
    a_sig = tuple(sorted(json.dumps(f, sort_keys=True) for f in a_features))
    b_sig = tuple(sorted(json.dumps(f, sort_keys=True) for f in b_features))
    return a_sig == b_sig


def collect_distinct_valid_plans(
    raw_votes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Compile all raw votes into distinct valid executable plans with vote support.

    Steps:
    1. Compile each draft and filter invalid plans.
    2. Group by full plan signature and count vote support.
    3. Sort groups by vote_support descending (stable tie-break by signature).
    4. Assign stable candidate_id (p1, p2, ...) to each distinct plan.
    5. Return representative plans enriched with candidate_id and vote_support.

    Returns:
        List of plan dicts, each with added keys:
        - "candidate_id": "p1", "p2", ...
        - "vote_support": int
    """
    if not raw_votes:
        return []

    # Compile and filter
    compiled: List[Dict[str, Any]] = []
    for vote in raw_votes:
        plan = compile_draft(vote)
        if _is_valid_executable_plan(plan):
            compiled.append(plan)

    if not compiled:
        return []

    # Group by signature and count votes
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for plan in compiled:
        sig = _plan_signature(plan)
        groups.setdefault(sig, []).append(plan)

    # Sort by vote_support descending, then by signature for stability
    sorted_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), str(item[0])),
    )

    out: List[Dict[str, Any]] = []
    for idx, (sig, group) in enumerate(sorted_groups, start=1):
        rep = group[0]
        plan: Dict[str, Any] = {
            "candidate_id": f"p{idx}",
            "vote_support": len(group),
            "status": rep.get("status", "ok"),
            "confidence": rep.get("confidence", 1.0),
            "tasks": rep.get("tasks", []),
            "constraints": _union_dicts([p.get("constraints", []) for p in group]),
        }
        if rep.get("reason") is not None:
            plan["reason"] = rep["reason"]
        out.append(plan)

    return out


def classify_plan_difference(
    primary_plan: Dict[str, Any],
    other_plan: Dict[str, Any],
) -> str:
    """
    Classify how two plans differ structurally.

    Returns one of:
    - "identical": same signature.
    - "different_task_count": different number of tasks.
    - "different_constraints": constraints differ.
    - "multiple_task_diffs": more than one task differs.
    - "single_task_diff": exactly one task differs (localizable).
    """
    if _plan_signature(primary_plan) == _plan_signature(other_plan):
        return "identical"

    primary_tasks = primary_plan.get("tasks", [])
    other_tasks = other_plan.get("tasks", [])

    if len(other_tasks) != len(primary_tasks):
        return "different_task_count"

    primary_constraints = tuple(
        sorted(json.dumps(c, sort_keys=True) for c in primary_plan.get("constraints", []))
    )
    other_constraints = tuple(
        sorted(json.dumps(c, sort_keys=True) for c in other_plan.get("constraints", []))
    )
    if primary_constraints != other_constraints:
        return "different_constraints"

    diffs = [
        i for i, (pt, ct) in enumerate(zip(primary_tasks, other_tasks))
        if not _task_eq(pt, ct)
    ]

    if len(diffs) == 0:
        return "identical"  # should not happen due to signature check
    if len(diffs) == 1:
        return "single_task_diff"
    return "multiple_task_diffs"


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
