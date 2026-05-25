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
) -> Tuple[bool, List[Dict[str, Any]], bool]:
    """
    Aggregate multiple LLM semantic-draft outputs into ranked compact plans.

    Strategy:
    1. Compile each draft into a compact plan.
    2. Group plans by their full plan signature.
    3. Rank groups by vote count descending.
    4. If top group has majority (> threshold), success=True.
    5. If no majority, success=False and needs_adjudication=True.

    Returns:
        (success, ranked_plans, needs_adjudication)
        ranked_plans is a list of compact plan dicts sorted by confidence.
    """
    if not raw_votes:
        return False, [], False

    # Compile drafts to compact plans
    compiled_plans: List[Dict[str, Any]] = []
    for vote in raw_votes:
        plan = compile_draft(vote)
        compiled_plans.append(plan)

    # Group by plan signature
    plan_groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for plan in compiled_plans:
        sig = _plan_signature(plan)
        plan_groups.setdefault(sig, []).append(plan)

    total_votes = len(raw_votes)
    majority_threshold = total_votes / 2

    # Sort groups by size descending
    sorted_groups = sorted(plan_groups.items(), key=lambda item: len(item[1]), reverse=True)

    needs_adjudication = len(sorted_groups[0][1]) <= majority_threshold

    # Build ranked plans
    ranked_plans: List[Dict[str, Any]] = []
    for sig, group in sorted_groups:
        # Aggregate fields within the group
        plan = _aggregate_plan_group(group, total_votes)
        ranked_plans.append(plan)

    success = not needs_adjudication or len(sorted_groups) == 1

    return success, ranked_plans, needs_adjudication


def _aggregate_plan_group(
    group: List[Dict[str, Any]],
    total_votes: int,
) -> Dict[str, Any]:
    """Aggregate a group of identical compact plans into a single representative plan."""
    representative = group[0]
    seq_len = len(representative.get("tasks", []))
    sequence_agreement = len(group) / total_votes

    tasks = []
    for step_idx in range(seq_len):
        step_tasks = []
        for plan in group:
            plan_tasks = plan.get("tasks", [])
            if step_idx < len(plan_tasks):
                step_tasks.append(plan_tasks[step_idx])

        best_task = _aggregate_step_task(step_tasks, step_idx + 1, sequence_agreement)
        tasks.append(best_task)

    # Union constraints
    constraints = _union_dicts([p.get("constraints", []) for p in group])

    # Confidence based on vote share
    confidence = len(group) / total_votes

    result: Dict[str, Any] = {
        "status": representative.get("status", "ok"),
        "confidence": round(confidence, 2),
        "tasks": tasks,
        "constraints": constraints,
    }
    if representative.get("reason") is not None:
        result["reason"] = representative["reason"]
    return result


def _mean_completeness(step_tasks: List[Dict[str, Any]]) -> float:
    """Average completeness score for a list of step tasks."""
    if not step_tasks:
        return 0.0
    scores = []
    for t in step_tasks:
        score = 0.0
        if t.get("action") and t.get("action") != "UNKNOWN":
            score += 0.5
        if t.get("features"):
            score += 0.3
        if t.get("direction"):
            score += 0.2
        scores.append(score)
    return sum(scores) / len(scores)


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


def _aggregate_step_task(
    step_tasks: List[Dict[str, Any]],
    step_id: int,
    sequence_agreement: float,
) -> Dict[str, Any]:
    """Aggregate multiple task parses for the same execution step."""
    total_votes = len(step_tasks)

    # action
    actions = [t.get("action", "UNKNOWN") for t in step_tasks if t.get("action")]
    action = Counter(actions).most_common(1)[0][0] if actions else "UNKNOWN"

    # direction
    directions = [t.get("direction") for t in step_tasks if t.get("direction")]
    direction = Counter(directions).most_common(1)[0][0] if directions else None

    # features: union by JSON key
    all_features = []
    for t in step_tasks:
        for f in t.get("features", []):
            all_features.append(f)
    features = _union_dicts([all_features])

    # model self-reported confidence
    model_confs = [t.get("confidence", 0.0) for t in step_tasks if t.get("confidence") is not None]
    mean_model_conf = sum(model_confs) / len(model_confs) if model_confs else 0.0

    # vote agreement on the winning action
    action_votes = sum(1 for t in step_tasks if t.get("action") == action)
    task_vote_agreement = action_votes / total_votes

    # score = weighted combination
    score = (
        0.50 * sequence_agreement
        + 0.25 * task_vote_agreement
        + 0.15 * mean_model_conf
        + 0.10 * _mean_completeness(step_tasks)
    )

    task: Dict[str, Any] = {
        "step_id": step_id,
        "action": action,
        "features": features,
        "confidence": round(score, 2),
    }
    if direction:
        task["direction"] = direction

    return task


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
