"""Output validation for compact parsed results."""

from typing import Dict, Any, List, Optional

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

VALID_STATUSES = {"ok", "needs_review", "unsupported", "none"}

VALID_FEATURE_ROLES = {
    "path",
    "where",
    "progress",
    "target",
    "terminate",
    "start",
}

VALID_CONSTRAINT_TYPES = {
    "forbidden_action",
}


def validate_result(result: Dict[str, Any]) -> None:
    """
    Validate the full compact parse result.

    Raises ValueError on structural or semantic violations.
    """
    status = result.get("status")
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")

    confidence = result.get("confidence")
    if not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be a number")
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(f"confidence out of range: {confidence}")

    tasks = result.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("tasks must be a list")

    for i, task in enumerate(tasks):
        _validate_task(task, i + 1)

    # Ensure continuous step_id starting from 1
    for i, task in enumerate(tasks):
        expected_id = i + 1
        if task.get("step_id") != expected_id:
            raise ValueError(
                f"step_id must be continuous starting from 1; expected {expected_id}, got {task.get('step_id')}"
            )

    constraints = result.get("constraints", [])
    if not isinstance(constraints, list):
        raise ValueError("constraints must be a list")
    for i, c in enumerate(constraints):
        _validate_constraint(c, i)

    alternatives = result.get("alternatives", [])
    if not isinstance(alternatives, list):
        raise ValueError("alternatives must be a list")
    # New protocol: alternatives must always be empty
    if alternatives:
        raise ValueError("alternatives must be empty in the new protocol")
    seen_ranks: set = set()
    prev_rank = 0
    for alt in alternatives:
        _validate_alternative_plan(alt)
        rank = alt.get("rank")
        if rank in seen_ranks:
            raise ValueError(f"Duplicate alternative rank: {rank}")
        seen_ranks.add(rank)
        if rank <= prev_rank:
            raise ValueError(f"Alternative ranks must be ascending; got {rank} after {prev_rank}")
        prev_rank = rank

    backtracking = result.get("backtracking", {})
    if not isinstance(backtracking, dict):
        raise ValueError("backtracking must be a dict")
    _validate_backtracking(backtracking, tasks)

    # Gate rules for status
    if status == "unsupported":
        if tasks:
            raise ValueError("status=unsupported must have empty tasks")
        if backtracking.get("step_candidates"):
            raise ValueError("status=unsupported must have empty backtracking.step_candidates")

    if status == "ok":
        if result.get("reason") is not None:
            raise ValueError("status=ok must not have a reason")
        for task in tasks:
            if task.get("action") == "UNKNOWN":
                raise ValueError("status=ok must not have UNKNOWN action")

    if status == "none":
        if tasks:
            raise ValueError("status=none should have empty tasks")
        if backtracking.get("step_candidates"):
            raise ValueError("status=none must have empty backtracking.step_candidates")


def _validate_task(task: Dict[str, Any], expected_step_id: int) -> None:
    """Validate a single compact task dict."""
    if not isinstance(task, dict):
        raise ValueError("Each task must be a dict")

    if task.get("step_id") != expected_step_id:
        raise ValueError(f"Expected step_id {expected_step_id}, got {task.get('step_id')}")

    action = task.get("action")
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid action: {action}")

    direction = task.get("direction")
    if direction is not None and direction not in VALID_DIRECTIONS:
        raise ValueError(f"Invalid direction: {direction}")

    conf = task.get("confidence")
    if conf is not None:
        if not isinstance(conf, (int, float)):
            raise ValueError("task confidence must be a number")
        if not (0.0 <= conf <= 1.0):
            raise ValueError(f"task confidence out of range: {conf}")

    features = task.get("features", [])
    if not isinstance(features, list):
        raise ValueError("features must be a list")
    for f in features:
        _validate_feature(f)


def _validate_feature(feature: Dict[str, Any]) -> None:
    """Validate a feature dict."""
    if not isinstance(feature, dict):
        raise ValueError("Each feature must be a dict")

    role = feature.get("role")
    if role not in VALID_FEATURE_ROLES:
        raise ValueError(f"Invalid feature role: {role}")

    if role in ("terminate", "start"):
        if not feature.get("trigger"):
            raise ValueError(f"Feature with role '{role}' must have a trigger")
        if not feature.get("landmark"):
            raise ValueError(f"Feature with role '{role}' must have a landmark")
    else:
        relation = feature.get("relation")
        if not relation:
            raise ValueError(f"Feature with role '{role}' must have a relation")
        if relation not in VALID_RELATIONS:
            raise ValueError(f"Invalid relation '{relation}' for feature role '{role}'")
        if not feature.get("landmark"):
            raise ValueError(f"Feature with role '{role}' must have a landmark")

    # Optional fields must not be empty strings if present
    for key in ("relation", "landmark", "trigger"):
        val = feature.get(key)
        if val is not None and val == "":
            raise ValueError(f"Feature field '{key}' must not be an empty string; omit it instead")


def _validate_constraint(constraint: Dict[str, Any], idx: int) -> None:
    """Validate a constraint dict."""
    if not isinstance(constraint, dict):
        raise ValueError("Each constraint must be a dict")

    ctype = constraint.get("type")
    if ctype not in VALID_CONSTRAINT_TYPES:
        raise ValueError(f"Invalid constraint type: {ctype}")

    action = constraint.get("action")
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid constraint action: {action}")

    direction = constraint.get("direction")
    if direction is not None and direction not in VALID_DIRECTIONS:
        raise ValueError(f"Invalid constraint direction: {direction}")

    features = constraint.get("features", [])
    if not isinstance(features, list):
        raise ValueError("constraint features must be a list")
    for f in features:
        _validate_feature(f)


def _validate_alternative_plan(alt: Dict[str, Any]) -> None:
    """Validate an alternative plan dict."""
    if not isinstance(alt, dict):
        raise ValueError("Each alternative must be a dict")

    rank = alt.get("rank")
    if rank not in (2, 3):
        raise ValueError("Alternative rank must be 2 or 3")

    conf = alt.get("confidence")
    if not isinstance(conf, (int, float)):
        raise ValueError("alternative confidence must be a number")
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"alternative confidence out of range: {conf}")

    alt_tasks = alt.get("tasks", [])
    if not isinstance(alt_tasks, list):
        raise ValueError("alternative tasks must be a list")
    for i, task in enumerate(alt_tasks):
        _validate_task(task, i + 1)

    for i, task in enumerate(alt_tasks):
        expected_id = i + 1
        if task.get("step_id") != expected_id:
            raise ValueError(
                f"alternative step_id must be continuous starting from 1; expected {expected_id}, got {task.get('step_id')}"
            )

    alt_constraints = alt.get("constraints", [])
    if not isinstance(alt_constraints, list):
        raise ValueError("alternative constraints must be a list")
    for i, c in enumerate(alt_constraints):
        _validate_constraint(c, i)


def _validate_backtracking(backtracking: Dict[str, Any], primary_tasks: List[Dict[str, Any]]) -> None:
    """Validate backtracking step_candidates."""
    step_candidates = backtracking.get("step_candidates", [])
    if not isinstance(step_candidates, list):
        raise ValueError("backtracking.step_candidates must be a list")

    primary_step_ids = {t.get("step_id") for t in primary_tasks}
    seen_group_step_ids: set = set()

    for group in step_candidates:
        if not isinstance(group, dict):
            raise ValueError("Each step_candidate group must be a dict")

        group_step_id = group.get("step_id")
        if group_step_id not in primary_step_ids:
            raise ValueError(f"step_candidate group step_id {group_step_id} not found in primary tasks")
        if group_step_id in seen_group_step_ids:
            raise ValueError(f"Duplicate step_candidate group for step_id {group_step_id}")
        seen_group_step_ids.add(group_step_id)

        candidates = group.get("candidates", [])
        if not isinstance(candidates, list):
            raise ValueError("step_candidate group candidates must be a list")

        seen_ranks: set = set()
        prev_rank = 0
        primary_task = next((t for t in primary_tasks if t.get("step_id") == group_step_id), None)

        for cand in candidates:
            _validate_backtracking_candidate(
                cand,
                group_step_id,
                primary_task,
            )
            rank = cand.get("rank")
            if rank not in (2, 3):
                raise ValueError(f"Backtracking candidate rank must be 2 or 3, got {rank}")
            if rank in seen_ranks:
                raise ValueError(f"Duplicate backtracking candidate rank {rank} in step {group_step_id}")
            seen_ranks.add(rank)
            if rank <= prev_rank:
                raise ValueError(
                    f"Backtracking candidate ranks must be ascending within step {group_step_id}; got {rank} after {prev_rank}"
                )
            prev_rank = rank


def _validate_backtracking_candidate(
    cand: Dict[str, Any],
    expected_step_id: int,
    primary_task: Optional[Dict[str, Any]],
) -> None:
    """Validate a single backtracking candidate dict."""
    if not isinstance(cand, dict):
        raise ValueError("Each backtracking candidate must be a dict")

    if cand.get("step_id") != expected_step_id:
        raise ValueError(
            f"Backtracking candidate step_id {cand.get('step_id')} does not match group step_id {expected_step_id}"
        )

    action = cand.get("action")
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid backtracking candidate action: {action}")

    direction = cand.get("direction")
    if direction is not None and direction not in VALID_DIRECTIONS:
        raise ValueError(f"Invalid backtracking candidate direction: {direction}")

    conf = cand.get("confidence")
    if not isinstance(conf, (int, float)):
        raise ValueError("backtracking candidate confidence must be a number")
    if not (0.0 <= conf <= 1.0):
        raise ValueError(f"backtracking candidate confidence out of range: {conf}")

    features = cand.get("features", [])
    if not isinstance(features, list):
        raise ValueError("backtracking candidate features must be a list")
    for f in features:
        _validate_feature(f)

    if primary_task is not None:
        if _task_eq(cand, primary_task):
            raise ValueError(
                f"Backtracking candidate for step {expected_step_id} must differ from primary task"
            )


def _task_eq(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Check whether two task dicts are identical for backtracking dedup."""
    if a.get("action") != b.get("action"):
        return False
    if a.get("direction") != b.get("direction"):
        return False
    if a.get("features") != b.get("features"):
        return False
    return True
