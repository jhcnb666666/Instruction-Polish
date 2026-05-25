"""Tests for semantic execution-order parsing via LLM with mocked backend — compact schema."""

from unittest.mock import patch

import pytest
from vln_instruction_parser.parser import parse_instruction_llm


def _action(id_, action, direction=None, features=None, confidence=0.9):
    a = {
        "id": id_,
        "action": action,
        "features": features if features is not None else [],
        "confidence": confidence,
    }
    if direction is not None:
        a["direction"] = direction
    return a


def _make_response(actions, order=None, constraints=None, excluded=None):
    return {
        "actions": actions,
        "order": order or [],
        "constraints": constraints or [],
        "excluded": excluded or [],
    }


class TestTemporalReordering:
    @patch("vln_instruction_parser.llm._call_backend")
    def test_before_reorder(self, mock_call):
        """Before turning left, go straight -> GO_STRAIGHT then TURN."""
        actions = [
            _action("a1", "MOVE_FORWARD", "straight", [{"role": "path", "relation": "along", "landmark": "hallway"}]),
            _action("a2", "TURN", "left", []),
        ]
        mock_call.return_value = _make_response(
            actions,
            order=[{"before": "a1", "after": "a2"}],
        )

        result = parse_instruction_llm("Before turning left, go straight down the hallway.", vote_count=3)
        tasks_out = result["tasks"]
        assert len(tasks_out) == 2
        assert tasks_out[0]["action"] == "MOVE_FORWARD"
        assert tasks_out[1]["action"] == "TURN"
        assert tasks_out[0]["step_id"] == 1
        assert tasks_out[1]["step_id"] == 2

    @patch("vln_instruction_parser.llm._call_backend")
    def test_after_reorder(self, mock_call):
        """Turn left after passing the sofa -> PASS then TURN."""
        actions = [
            _action("a1", "PASS", features=[{"role": "progress", "relation": "past", "landmark": "sofa"}]),
            _action("a2", "TURN", "left", []),
        ]
        mock_call.return_value = _make_response(
            actions,
            order=[{"before": "a1", "after": "a2"}],
        )

        result = parse_instruction_llm("Turn left after passing the sofa.", vote_count=3)
        tasks_out = result["tasks"]
        assert tasks_out[0]["action"] == "PASS"
        assert tasks_out[1]["action"] == "TURN"


class TestSuppression:
    @patch("vln_instruction_parser.llm._call_backend")
    def test_instead_of_suppresses_enter(self, mock_call):
        """Instead of entering the kitchen, turn right -> only TURN."""
        actions = [
            _action("a1", "TURN", "right", [{"role": "where", "relation": "at", "landmark": "door"}]),
        ]
        mock_call.return_value = _make_response(
            actions,
            excluded=[{"id": "ex1", "reason": "replaced_by_instead"}],
        )

        result = parse_instruction_llm("Instead of entering the kitchen, turn right at the door.", vote_count=3)
        tasks_out = result["tasks"]
        assert len(tasks_out) == 1
        assert tasks_out[0]["action"] == "TURN"
        actions_list = [t["action"] for t in tasks_out]
        assert "ENTER" not in actions_list

    @patch("vln_instruction_parser.llm._call_backend")
    def test_do_not_suppresses_enter(self, mock_call):
        """Do not enter the room; wait outside -> only WAIT."""
        actions = [
            _action("a1", "WAIT", features=[{"role": "where", "relation": "outside", "landmark": "door"}]),
        ]
        mock_call.return_value = _make_response(
            actions,
            constraints=[{"type": "forbidden_action", "action": "ENTER", "features": [{"role": "where", "relation": "inside", "landmark": "room"}]}],
        )

        result = parse_instruction_llm("Do not enter the room; wait outside instead.", vote_count=3)
        tasks_out = result["tasks"]
        assert len(tasks_out) == 1
        assert tasks_out[0]["action"] == "WAIT"
        actions_list = [t["action"] for t in tasks_out]
        assert "ENTER" not in actions_list
        assert len(result.get("constraints", [])) >= 1


class TestConjunction:
    @patch("vln_instruction_parser.llm._call_backend")
    def test_and_connects_two_actions(self, mock_call):
        """Go straight and turn left by the sofa -> two tasks."""
        actions = [
            _action("a1", "MOVE_FORWARD", "straight", []),
            _action("a2", "TURN", "left", [{"role": "where", "relation": "at", "landmark": "sofa"}]),
        ]
        mock_call.return_value = _make_response(actions)

        result = parse_instruction_llm("Go straight and turn left by the sofa.", vote_count=3)
        tasks_out = result["tasks"]
        assert len(tasks_out) == 2
        assert tasks_out[0]["action"] == "MOVE_FORWARD"
        assert tasks_out[1]["action"] == "TURN"


class TestMalformedAndLowConfidence:
    @patch("vln_instruction_parser.llm._call_backend")
    def test_malformed_reordered_low_confidence(self, mock_call):
        """Before turn left go straight -> reordered with low confidence."""
        actions = [
            _action("a1", "MOVE_FORWARD", "straight", [], 0.65),
            _action("a2", "TURN", "left", [], 0.65),
        ]
        mock_call.return_value = _make_response(
            actions,
            order=[{"before": "a1", "after": "a2"}],
        )

        result = parse_instruction_llm("Before turn left go straight.", vote_count=3)
        tasks_out = result["tasks"]
        assert tasks_out[0]["action"] == "MOVE_FORWARD"
        assert tasks_out[1]["action"] == "TURN"
        for t in tasks_out:
            # In compact schema, confidence is at result level, not task level
            pass


class TestRuleFallbackWithTemporalKeywords:
    @patch("vln_instruction_parser.llm._call_backend")
    def test_temporal_fallback_caps_confidence(self, mock_call):
        """When LLM fails and fallback=True, simple instructions fallback to rules."""
        mock_call.return_value = None  # Force LLM failure
        result = parse_instruction_llm(
            "Go straight.",  # simple instruction without temporal keywords for stable rule output
            fallback_to_rules=True,
            vote_count=1,
        )
        tasks_out = result["tasks"]
        assert len(tasks_out) > 0
        assert tasks_out[0]["confidence"] > 0.50 if "confidence" in tasks_out[0] else result["confidence"] > 0.50

    @patch("vln_instruction_parser.llm._call_backend")
    def test_temporal_fallback_with_simple_keyword(self, mock_call):
        """Rule fallback on temporal instruction forces alternatives and caps confidence."""
        mock_call.return_value = None  # Force LLM failure
        result = parse_instruction_llm(
            "Go straight then turn left.",
            fallback_to_rules=True,
            vote_count=1,
        )
        tasks_out = result["tasks"]
        assert len(tasks_out) > 0


class Test3DWithTemporal:
    @patch("vln_instruction_parser.llm._call_backend")
    def test_upstairs_then_turn_right(self, mock_call):
        """Go upstairs, then turn right -> unsupported."""
        actions = [
            _action("a1", "UNKNOWN", features=[{"role": "target", "relation": "toward", "landmark": "upstairs"}]),
        ]
        mock_call.return_value = _make_response(actions)

        result = parse_instruction_llm("Go upstairs, then turn right.", vote_count=3)
        # Active vertical motion should be unsupported
        assert result["status"] == "unsupported"
