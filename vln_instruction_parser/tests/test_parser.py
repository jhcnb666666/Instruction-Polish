"""End-to-end parser tests — compact schema."""

import pytest
from unittest.mock import patch, MagicMock
from vln_instruction_parser.parser import parse_instruction, parse_instruction_llm


# Context manager to bypass the complexity gate so we can test rule parsing directly
_bypass_complexity = patch("vln_instruction_parser.complexity.requires_semantic_parser", return_value=False)


class TestParserBasic:
    def test_empty_string(self):
        result = parse_instruction("")
        assert result["status"] == "ok"
        assert result["confidence"] == 1.0
        assert result["tasks"] == []

    def test_non_english(self):
        result = parse_instruction("走到厨房")
        assert result["tasks"] == []

    def test_single_segment(self):
        result = parse_instruction("Go to the kitchen")
        assert len(result["tasks"]) == 1
        task = result["tasks"][0]
        assert task["step_id"] == 1
        assert task["action"] == "GO_TO"
        # Compact schema: features list, not flat landmark
        assert any(f.get("landmark") == "kitchen" for f in task.get("features", []))
        assert 0 <= result["confidence"] <= 1.0

    def test_complexity_gate_blocks_complex(self):
        """Complex instructions (3+ actions) are rejected by the rule parser."""
        text = "Go down the hallway, turn left in front of the sofa, then stop by the kitchen door."
        result = parse_instruction(text)
        assert result["status"] == "needs_review"
        assert result["tasks"] == []


class TestParserAcceptanceCases:
    # These tests exercise the rule parser on multi-action instructions.
    # We bypass the complexity gate because we are testing rule-parsing capability,
    # not the routing logic (which is tested separately).

    def test_case_1_hallway_sofa_kitchen(self):
        text = "Go down the hallway, turn left in front of the sofa, then stop by the kitchen door."
        with _bypass_complexity:
            result = parse_instruction(text)
        assert len(result["tasks"]) == 3

        t1 = result["tasks"][0]
        assert t1["step_id"] == 1
        assert t1["action"] == "MOVE_FORWARD"
        assert any("hallway" in (f.get("landmark") or "") for f in t1.get("features", []))

        t2 = result["tasks"][1]
        assert t2["step_id"] == 2
        assert t2["action"] == "TURN"
        assert t2["direction"] == "left"
        assert any("sofa" in (f.get("landmark") or "") for f in t2.get("features", []))

        t3 = result["tasks"][2]
        assert t3["step_id"] == 3
        assert t3["action"] == "STOP"
        assert any("kitchen" in (f.get("landmark") or "") for f in t3.get("features", []))

    def test_case_2_walk_turn_stop(self):
        text = "Walk to the table. Turn right. Stop."
        with _bypass_complexity:
            result = parse_instruction(text)
        assert len(result["tasks"]) == 3

        t1 = result["tasks"][0]
        assert t1["action"] == "MOVE_FORWARD"
        assert any("table" in (f.get("landmark") or "") for f in t1.get("features", []))

        t2 = result["tasks"][1]
        assert t2["action"] == "TURN"
        assert t2["direction"] == "right"
        # NO context inheritance per spec
        assert not any(f.get("landmark") for f in t2.get("features", []))

        t3 = result["tasks"][2]
        assert t3["action"] == "STOP"
        # STOP segment is just "Stop"; the word "stop" itself may be extracted as landmark
        # which is acceptable since it appears in the instruction text

    def test_case_3_straight_past_enter(self):
        text = "Go straight past the couch and enter the bedroom."
        result = parse_instruction(text)
        assert len(result["tasks"]) >= 2

        actions = [t["action"] for t in result["tasks"]]
        assert "MOVE_FORWARD" in actions or "GO_TO" in actions
        assert "ENTER" in actions

        enter_task = next(t for t in result["tasks"] if t["action"] == "ENTER")
        assert any("bedroom" in (f.get("landmark") or "") for f in enter_task.get("features", []))

    def test_case_4_between_face(self):
        text = "Move between the two chairs, then face the door."
        result = parse_instruction(text)
        assert len(result["tasks"]) == 2

        t1 = result["tasks"][0]
        assert t1["action"] == "MOVE_FORWARD"
        assert any("chairs" in (f.get("landmark") or "") for f in t1.get("features", []))

        t2 = result["tasks"][1]
        assert t2["action"] == "FACE"
        assert any("door" in (f.get("landmark") or "") for f in t2.get("features", []))

    def test_case_5_go_over_there_wait(self):
        text = "Go over there and wait."
        result = parse_instruction(text)
        assert len(result["tasks"]) == 2

        t1 = result["tasks"][0]
        assert t1["action"] in ("GO_TO", "MOVE_FORWARD")

        t2 = result["tasks"][1]
        assert t2["action"] == "WAIT"

    def test_case_6_end_turn_around_back(self):
        text = "Walk to the end of the hallway, turn around, and go back to the entrance."
        with _bypass_complexity:
            result = parse_instruction(text)
        assert len(result["tasks"]) == 3

        t1 = result["tasks"][0]
        assert t1["action"] == "MOVE_FORWARD"
        assert any("hallway" in (f.get("landmark") or "") for f in t1.get("features", []))

        t2 = result["tasks"][1]
        assert t2["action"] == "TURN"
        assert t2["direction"] == "around"

        t3 = result["tasks"][2]
        assert t3["action"] in ("GO_TO", "MOVE_FORWARD")
        assert t3["direction"] == "backward"
        assert any("entrance" in (f.get("landmark") or "") for f in t3.get("features", []))

    def test_case_7_3d_upstairs(self):
        text = "Go upstairs, turn right, and stop outside the office."
        result = parse_instruction(text)
        assert result["status"] == "unsupported"
        assert result["tasks"] == []
        assert result.get("reason") == "vertical_motion_not_supported"


class TestParserNoTextInOutput:
    def test_no_original_instruction(self):
        result = parse_instruction("Turn left.")
        assert "original_instruction" not in result
        assert "raw_text" not in result
        assert "canonical_instruction" not in result

    def test_no_source_order(self):
        result = parse_instruction("Go straight.")
        for t in result["tasks"]:
            assert "source_order" not in t


class TestParserContinuousStepIds:
    def test_continuous_ids(self):
        text = "Go to the kitchen. Turn left. Stop. Enter the room."
        with _bypass_complexity:
            result = parse_instruction(text)
        for i, task in enumerate(result["tasks"], start=1):
            assert task["step_id"] == i


class TestParser2DOnly:
    def test_no_up_down_directions(self):
        text = "Go to the kitchen"
        result = parse_instruction(text)
        for task in result["tasks"]:
            if task.get("direction"):
                assert task["direction"] not in ("up", "down", "upstairs", "downstairs")
            assert task["action"] != "climb"


class TestParserLLMFallback:
    @patch("vln_instruction_parser.llm._call_backend")
    def test_fallback_on_backend_error(self, mock_call):
        """When LLM backend fails and fallback=True, simple instructions fallback to rules."""
        mock_call.return_value = None
        text = "Walk to the table and stop."
        result = parse_instruction_llm(text, fallback_to_rules=True)
        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["action"] == "MOVE_FORWARD"

    @patch("vln_instruction_parser.llm._call_backend")
    def test_no_fallback_returns_empty(self, mock_call):
        """When LLM backend fails and fallback=False, return empty tasks."""
        mock_call.return_value = None
        text = "Walk to the table."
        result = parse_instruction_llm(text, fallback_to_rules=False)
        assert result["tasks"] == []
        assert result["reason"] == "initial_generation_failed"

    @patch("vln_instruction_parser.llm._call_backend")
    def test_complex_fail_closed_no_fallback(self, mock_call):
        """Complex instructions fail closed when LLM is unavailable."""
        mock_call.return_value = None
        text = "Before turning left, go straight down the hallway."
        result = parse_instruction_llm(text, fallback_to_rules=True)
        assert result["status"] == "needs_review"
        assert result["tasks"] == []
        assert result["reason"] == "initial_generation_failed"
