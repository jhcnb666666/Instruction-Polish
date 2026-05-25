"""Tests for step feature extraction and pruning behavior."""

from unittest.mock import patch

import pytest
from vln_instruction_parser.parser import parse_instruction, parse_instruction_llm


_bypass_complexity = patch("vln_instruction_parser.complexity.requires_semantic_parser", return_value=False)


class TestStepFeatures:
    def test_turn_left_no_fabricated_location(self):
        result = parse_instruction("Turn left.")
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["features"] == []
        assert "original_instruction" not in result

    def test_go_straight_until_door_terminate(self):
        """Go straight until you reach the door -> terminate feature (LLM path)."""
        from unittest.mock import patch
        with patch("vln_instruction_parser.llm._call_backend") as mock_call:
            mock_call.return_value = {
                "actions": [
                    {"id": "a1", "action": "MOVE_FORWARD", "direction": "straight", "features": [
                        {"role": "terminate", "trigger": "reach", "landmark": "door"}
                    ], "confidence": 0.9}
                ],
                "order": [],
                "constraints": [],
                "excluded": [],
            }
            result = parse_instruction_llm("Go straight until you reach the door.", vote_count=3)
        t = result["tasks"][0]
        terminate = next((f for f in t.get("features", []) if f["role"] == "terminate"), None)
        assert terminate is not None
        assert terminate["landmark"] == "door"

    def test_when_see_sofa_turn_left_start(self):
        """When you see the sofa, turn left -> start feature (LLM path)."""
        from unittest.mock import patch
        with patch("vln_instruction_parser.llm._call_backend") as mock_call:
            mock_call.return_value = {
                "actions": [
                    {"id": "a1", "action": "TURN", "direction": "left", "features": [
                        {"role": "start", "trigger": "see", "landmark": "sofa"}
                    ], "confidence": 0.9}
                ],
                "order": [],
                "constraints": [],
                "excluded": [],
            }
            result = parse_instruction_llm("When you see the sofa, turn left.", vote_count=3)
        t = result["tasks"][0]
        start = next((f for f in t.get("features", []) if f["role"] == "start"), None)
        assert start is not None
        assert start["landmark"] == "sofa"

    def test_follow_hallway_until_sofa_then_turn(self):
        """Adjacent steps with different role for same landmark (LLM path)."""
        from unittest.mock import patch
        with patch("vln_instruction_parser.llm._call_backend") as mock_call:
            mock_call.return_value = {
                "actions": [
                    {"id": "a1", "action": "MOVE_FORWARD", "direction": "straight", "features": [
                        {"role": "path", "relation": "along", "landmark": "hallway"},
                        {"role": "terminate", "trigger": "see", "relation": "left_of_agent", "landmark": "sofa"}
                    ], "confidence": 0.9},
                    {"id": "a2", "action": "TURN", "direction": "left", "features": [
                        {"role": "where", "relation": "at", "landmark": "sofa"}
                    ], "confidence": 0.9}
                ],
                "order": [{"before": "a1", "after": "a2"}],
                "constraints": [],
                "excluded": [],
            }
            result = parse_instruction_llm("Follow the hallway until you see the sofa on your left, then turn left at the sofa.", vote_count=3)
        t1 = result["tasks"][0]
        t2 = result["tasks"][1]
        t1_terminate = next((f for f in t1.get("features", []) if f["role"] == "terminate" and f["landmark"] == "sofa"), None)
        t2_where = next((f for f in t2.get("features", []) if f["role"] == "where" and f["landmark"] == "sofa"), None)
        assert t1_terminate is not None
        assert t2_where is not None

    def test_before_turning_right_walk_past_table_order(self):
        """Before turning right at the door, walk past the table."""
        from unittest.mock import patch
        with patch("vln_instruction_parser.llm._call_backend") as mock_call:
            mock_call.return_value = {
                "actions": [
                    {"id": "a1", "action": "PASS", "features": [
                        {"role": "progress", "relation": "past", "landmark": "table"}
                    ], "confidence": 0.9},
                    {"id": "a2", "action": "TURN", "direction": "right", "features": [
                        {"role": "where", "relation": "at", "landmark": "door"}
                    ], "confidence": 0.9}
                ],
                "order": [{"before": "a1", "after": "a2"}],
                "constraints": [],
                "excluded": [],
            }
            result = parse_instruction_llm("Before turning right at the door, walk past the table.", vote_count=3)
        assert result["tasks"][0]["action"] == "PASS"
        assert result["tasks"][1]["action"] == "TURN"
        assert "ordering_constraints" not in result
        assert "before" not in str(result)

    def test_walk_to_table_turn_right_no_inheritance(self):
        result = parse_instruction("Walk to the table. Turn right.")
        t2 = result["tasks"][1]
        assert t2["action"] == "TURN"
        assert not any(f.get("landmark") for f in t2.get("features", []))

    def test_walk_past_stairs_normal_2d(self):
        result = parse_instruction("Walk past the stairs.")
        assert result["status"] == "ok"
        assert all(t["action"] != "UNKNOWN" for t in result["tasks"])

    def test_walk_downstairs_unsupported(self):
        result = parse_instruction("Walk downstairs and turn left.")
        assert result["status"] == "unsupported"
        assert result.get("reason") == "vertical_motion_not_supported"
