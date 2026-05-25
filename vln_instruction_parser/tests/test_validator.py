"""Tests for compact output validation."""

import pytest
from vln_instruction_parser.validator import validate_result


def make_valid_task(
    step_id=1,
    action="MOVE_FORWARD",
    direction=None,
    features=None,
):
    features = features if features is not None else []
    task = {
        "step_id": step_id,
        "action": action,
        "features": features,
    }
    if direction is not None:
        task["direction"] = direction
    return task


def make_feature(role="where", relation="at", landmark="sofa", trigger=None):
    f = {"role": role, "relation": relation, "landmark": landmark}
    if trigger is not None:
        f["trigger"] = trigger
    return f


def make_alt(rank=2, confidence=0.84, tasks=None, constraints=None):
    return {
        "rank": rank,
        "confidence": confidence,
        "tasks": tasks if tasks is not None else [],
        "constraints": constraints if constraints is not None else [],
    }


class TestValidateResult:
    def test_valid_tier1_no_alternatives(self):
        result = {
            "status": "ok",
            "confidence": 1.0,
            "tasks": [make_valid_task()],
            "constraints": [],
            "alternatives": [],
        }
        validate_result(result)

    def test_valid_with_features(self):
        result = {
            "status": "ok",
            "confidence": 1.0,
            "tasks": [
                make_valid_task(
                    action="MOVE_FORWARD",
                    features=[make_feature(role="path", relation="along", landmark="hallway")],
                )
            ],
            "constraints": [],
            "alternatives": [],
        }
        validate_result(result)

    def test_valid_terminate_feature(self):
        result = {
            "status": "ok",
            "confidence": 1.0,
            "tasks": [
                make_valid_task(
                    action="MOVE_FORWARD",
                    features=[make_feature(role="terminate", relation="left_of_agent", landmark="sofa", trigger="see")],
                )
            ],
            "constraints": [],
            "alternatives": [],
        }
        validate_result(result)

    def test_valid_with_alternatives(self):
        result = {
            "status": "needs_review",
            "confidence": 0.93,
            "tasks": [make_valid_task()],
            "constraints": [],
            "alternatives": [
                make_alt(
                    rank=2,
                    confidence=0.84,
                    tasks=[make_valid_task(action="TURN", direction="left")],
                )
            ],
        }
        validate_result(result)

    def test_ok_with_alternatives_raises(self):
        result = {
            "status": "ok",
            "confidence": 1.0,
            "tasks": [make_valid_task()],
            "constraints": [],
            "alternatives": [make_alt()],
        }
        with pytest.raises(ValueError, match="status=ok must not have alternatives"):
            validate_result(result)

    def test_bad_step_id(self):
        result = {
            "status": "ok",
            "confidence": 1.0,
            "tasks": [make_valid_task(step_id=2)],
            "constraints": [],
            "alternatives": [],
        }
        with pytest.raises(ValueError):
            validate_result(result)

    def test_invalid_action(self):
        result = {
            "status": "ok",
            "confidence": 1.0,
            "tasks": [make_valid_task(action="FLY")],
            "constraints": [],
            "alternatives": [],
        }
        with pytest.raises(ValueError, match="Invalid action"):
            validate_result(result)

    def test_invalid_direction(self):
        result = {
            "status": "ok",
            "confidence": 1.0,
            "tasks": [make_valid_task(direction="up")],
            "constraints": [],
            "alternatives": [],
        }
        with pytest.raises(ValueError, match="Invalid direction"):
            validate_result(result)

    def test_invalid_feature_role(self):
        result = {
            "status": "ok",
            "confidence": 1.0,
            "tasks": [make_valid_task(features=[{"role": "invalid"}])],
            "constraints": [],
            "alternatives": [],
        }
        with pytest.raises(ValueError, match="Invalid feature role"):
            validate_result(result)

    def test_terminate_without_trigger(self):
        result = {
            "status": "ok",
            "confidence": 1.0,
            "tasks": [make_valid_task(features=[{"role": "terminate", "landmark": "sofa"}])],
            "constraints": [],
            "alternatives": [],
        }
        with pytest.raises(ValueError, match="trigger"):
            validate_result(result)

    def test_where_without_relation(self):
        result = {
            "status": "ok",
            "confidence": 1.0,
            "tasks": [make_valid_task(features=[{"role": "where", "landmark": "sofa"}])],
            "constraints": [],
            "alternatives": [],
        }
        with pytest.raises(ValueError, match="relation"):
            validate_result(result)

    def test_confidence_out_of_range(self):
        result = {
            "status": "ok",
            "confidence": 1.5,
            "tasks": [],
            "constraints": [],
            "alternatives": [],
        }
        with pytest.raises(ValueError, match="confidence out of range"):
            validate_result(result)

    def test_unsupported_with_tasks_raises(self):
        result = {
            "status": "unsupported",
            "confidence": 1.0,
            "tasks": [make_valid_task()],
            "constraints": [],
            "alternatives": [],
        }
        with pytest.raises(ValueError, match="unsupported must have empty tasks"):
            validate_result(result)

    def test_valid_unsupported(self):
        result = {
            "status": "unsupported",
            "confidence": 1.0,
            "tasks": [],
            "constraints": [],
            "alternatives": [],
            "reason": "vertical_motion_not_supported",
        }
        validate_result(result)
