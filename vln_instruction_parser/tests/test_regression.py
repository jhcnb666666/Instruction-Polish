"""Regression tests for identified failure paths — compact schema."""

from unittest.mock import patch

import pytest
from vln_instruction_parser.parser import parse_instruction_llm, parse_instruction
from vln_instruction_parser.validator import validate_result


def _task(step_id, action, direction=None, features=None, confidence=0.9):
    t = {
        "step_id": step_id,
        "action": action,
        "features": features if features is not None else [],
    }
    if direction is not None:
        t["direction"] = direction
    return t


def _wrap(actions):
    return {
        "actions": actions,
        "order": [],
        "constraints": [],
        "excluded": [],
    }


class TestRuleFallbackTemporal:
    """Complex temporal/negation inputs are rejected by the rule parser (fail-closed)."""

    def test_before_turning_left_go_straight(self):
        result = parse_instruction("Before turning left, go straight down the hallway.")
        assert result["status"] == "needs_review"
        assert result["tasks"] == []

    def test_instead_of_entering_turn_right(self):
        result = parse_instruction("Instead of entering the kitchen, turn right at the door.")
        assert result["status"] == "needs_review"
        assert result["tasks"] == []

    def test_do_not_enter_wait_outside(self):
        result = parse_instruction("Do not enter the room; wait outside instead.")
        assert result["status"] == "needs_review"
        assert result["tasks"] == []

    @patch("vln_instruction_parser.llm._call_backend")
    def test_llm_fallback_temporal_returns_needs_review(self, mock_call):
        """LLM failure on temporal input must fail-closed with needs_review."""
        mock_call.return_value = None
        result = parse_instruction_llm(
            "Before turning left, go straight down the hallway.",
            fallback_to_rules=True,
            vote_count=1,
        )
        assert result["status"] == "needs_review"
        assert result["tasks"] == []


class TestLocalAdjudication:
    """Adjudication must receive actual conflict candidates."""

    def test_adjudication_prompt_contains_candidates(self):
        """Build adjudication prompt and verify candidate plans are present."""
        from vln_instruction_parser.llm import _build_adjudication_prompt
        candidates = [
            [{"action": "MOVE_FORWARD", "direction": "straight", "features": []},
             {"action": "TURN", "direction": "left", "features": []}],
            [{"action": "TURN", "direction": "left", "features": []},
             {"action": "MOVE_FORWARD", "direction": "straight", "features": []}],
        ]
        prompt = _build_adjudication_prompt("Test instruction", candidates)
        assert "Candidate 1" in prompt
        assert "Candidate 2" in prompt
        assert "MOVE_FORWARD" in prompt
        assert "TURN" in prompt


class TestSequenceConflict:
    """Direction-level sequence conflicts must trigger adjudication."""

    def test_turn_left_then_right_vs_turn_right_then_left(self):
        vote_a = _wrap([
            _task(1, "TURN", "left", [{"role": "where", "relation": "at", "landmark": "sofa"}]),
            _task(2, "TURN", "right", [{"role": "where", "relation": "at", "landmark": "door"}]),
        ])
        vote_b = _wrap([
            _task(1, "TURN", "left", [{"role": "where", "relation": "at", "landmark": "sofa"}]),
            _task(2, "TURN", "right", [{"role": "where", "relation": "at", "landmark": "door"}]),
        ])
        vote_c = _wrap([
            _task(1, "TURN", "right", [{"role": "where", "relation": "at", "landmark": "door"}]),
            _task(2, "TURN", "left", [{"role": "where", "relation": "at", "landmark": "sofa"}]),
        ])
        from vln_instruction_parser.aggregator import aggregate_votes
        ranked = aggregate_votes([vote_a, vote_b, vote_c], "test")
        # 2/3 agree on one sequence, so it is the top candidate
        assert len(ranked) >= 1
        assert ranked[0]["vote_support"] == 2

    def test_turn_swapped_directions_keeps_all_candidates(self):
        vote_a = _wrap([
            _task(1, "TURN", "left", [{"role": "where", "relation": "at", "landmark": "sofa"}]),
            _task(2, "TURN", "right", [{"role": "where", "relation": "at", "landmark": "door"}]),
        ])
        vote_b = _wrap([
            _task(1, "TURN", "right", [{"role": "where", "relation": "at", "landmark": "door"}]),
            _task(2, "TURN", "left", [{"role": "where", "relation": "at", "landmark": "sofa"}]),
        ])
        vote_c = _wrap([
            _task(1, "TURN", "left", [{"role": "where", "relation": "at", "landmark": "window"}]),
            _task(2, "TURN", "right", [{"role": "where", "relation": "at", "landmark": "door"}]),
        ])
        from vln_instruction_parser.aggregator import aggregate_votes
        ranked = aggregate_votes([vote_a, vote_b, vote_c], "test")
        # All three votes are different plans, so all three are kept
        assert len(ranked) == 3


class TestNormalizations:
    """Specific normalization edge cases."""

    def test_stop_beside_the_sofa(self):
        result = parse_instruction("Stop beside the sofa.")
        validate_result(result)
        assert result["tasks"][0]["action"] == "STOP"

    def test_walk_past_the_floor_lamp(self):
        result = parse_instruction("Walk past the floor lamp, then stop by the door.")
        validate_result(result)
        for t in result["tasks"]:
            assert t["action"] != "UNKNOWN"


class TestSchemaRoundtrip:
    """Schema dataclass must round-trip through dict and validation."""

    def test_task_to_dict_then_validate(self):
        from vln_instruction_parser.schema import Task, Feature, ParseResult, task_to_dict, result_to_dict
        task = Task(
            step_id=1,
            action="TURN",
            direction="left",
            features=[Feature(role="where", relation="at", landmark="sofa")],
        )
        d = task_to_dict(task)
        assert "direction" in d
        assert "features" in d
        assert "source_order" not in d

        result = ParseResult(
            status="ok",
            confidence=1.0,
            tasks=[task],
        )
        result_dict = result_to_dict(result)
        validate_result(result_dict)
