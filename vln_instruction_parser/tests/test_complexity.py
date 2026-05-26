"""Tests for complexity gating and parse_instruction_auto routing."""

from unittest.mock import patch

from vln_instruction_parser.complexity import requires_semantic_parser
from vln_instruction_parser.parser import parse_instruction_auto, parse_instruction


class TestComplexityDetection:
    def test_temporal_before(self):
        assert requires_semantic_parser("Before turning left, go straight.")

    def test_temporal_after(self):
        assert requires_semantic_parser("Turn left after passing the sofa.")

    def test_negation_do_not(self):
        assert requires_semantic_parser("Do not enter the room.")

    def test_instead_of(self):
        assert requires_semantic_parser("Instead of entering the kitchen, go straight.")

    def test_phrasal_follow(self):
        assert requires_semantic_parser("Follow the hallway to the end.")

    def test_phrasal_take_a_left(self):
        assert requires_semantic_parser("Take a left at the door.")

    def test_semicolon(self):
        assert requires_semantic_parser("Go straight; turn left.")

    def test_three_actions(self):
        assert requires_semantic_parser("Go straight, turn left, then stop.")

    def test_stairs_active_vertical(self):
        assert requires_semantic_parser("Go down the stairs and stop.")

    def test_elevator_active_vertical(self):
        assert requires_semantic_parser("Take the elevator up to the second floor.")

    def test_walk_past_stairs_not_complex(self):
        """Passive reference to stairs should NOT trigger semantic parser."""
        assert not requires_semantic_parser("Walk past the stairs.")

    def test_stop_beside_elevator_not_complex(self):
        """Passive reference to elevator should NOT trigger semantic parser."""
        assert not requires_semantic_parser("Stop beside the elevator.")

    def test_simple_two_actions_not_complex(self):
        assert not requires_semantic_parser("Go straight and stop.")

    def test_simple_turn_not_complex(self):
        assert not requires_semantic_parser("Turn left in front of the sofa.")


class TestParseInstructionAuto:
    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_routes_simple_to_llm(self, mock_llm):
        """Simple instructions now go through LLM-first pipeline."""
        mock_llm.return_value = {
            "status": "ok",
            "confidence": 1.0,
            "tasks": [],
            "constraints": [],
            "alternatives": [],
        }
        result = parse_instruction_auto("Turn left.")
        mock_llm.assert_called_once()
        assert result["status"] == "ok"

    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_llm_result_passed_through(self, mock_llm):
        """parse_instruction_auto returns whatever parse_instruction_llm returns."""
        mock_llm.return_value = {
            "status": "needs_review",
            "confidence": 0.87,
            "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": [], "confidence": 0.85}],
            "constraints": [],
            "alternatives": [],
        }
        result = parse_instruction_auto("Turn left.")
        mock_llm.assert_called_once()
        assert result["status"] == "needs_review"
        assert result["confidence"] == 0.87

    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_routes_complex_to_llm(self, mock_llm):
        """Complex instructions should be routed to LLM."""
        mock_llm.return_value = {
            "status": "ok",
            "confidence": 1.0,
            "tasks": [],
            "constraints": [],
            "alternatives": [],
        }
        result = parse_instruction_auto("Before turning left, go straight.")
        mock_llm.assert_called_once()

    def test_complex_parse_instruction_returns_needs_review(self):
        """parse_instruction directly on complex input returns needs_review."""
        result = parse_instruction("Before turning left, go straight.")
        assert result["status"] == "needs_review"
        assert result["tasks"] == []
