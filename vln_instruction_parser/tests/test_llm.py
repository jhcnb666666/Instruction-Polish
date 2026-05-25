"""Tests for LLM provider module with mocked backend — compact semantic drafts."""

import json
from unittest.mock import MagicMock, patch

import pytest
from vln_instruction_parser.llm import parse_with_llm


def _make_action(id_, action, direction=None, features=None, confidence=0.9):
    a = {
        "id": id_,
        "action": action,
        "features": features if features is not None else [],
        "confidence": confidence,
    }
    if direction is not None:
        a["direction"] = direction
    return a


class TestParseWithLLM:
    @patch("vln_instruction_parser.llm._call_backend")
    def test_three_votes_consistent(self, mock_call):
        """All three votes return identical valid compact JSON."""
        actions = [
            _make_action("a1", "MOVE_FORWARD", "straight", [{"role": "path", "relation": "along", "landmark": "hallway"}]),
            _make_action("a2", "TURN", "left", [{"role": "where", "relation": "at", "landmark": "sofa"}]),
        ]
        mock_call.return_value = {
            "actions": actions,
            "order": [],
            "constraints": [],
            "excluded": [],
        }

        success, votes = parse_with_llm(
            "Go down the hallway, turn left in front of the sofa",
            vote_count=3,
        )
        assert success is True
        assert len(votes) == 3
        for v in votes:
            assert len(v["actions"]) == 2
            assert v["actions"][0]["action"] == "MOVE_FORWARD"
            assert v["actions"][1]["action"] == "TURN"

    @patch("vln_instruction_parser.llm._call_backend")
    def test_llm_returns_invalid_json(self, mock_call):
        """Backend returns non-JSON content -> success=False."""
        mock_call.return_value = None

        success, votes = parse_with_llm("test", vote_count=1)
        assert success is False
        assert votes == []

    @patch("vln_instruction_parser.llm._call_backend")
    def test_3d_instruction_low_confidence(self, mock_call):
        """LLM should return UNKNOWN or low confidence for 3D phrases."""
        actions = [_make_action("a1", "UNKNOWN", features=[], confidence=0.1)]
        mock_call.return_value = {
            "actions": actions,
            "order": [],
            "constraints": [],
            "excluded": [],
        }

        success, votes = parse_with_llm("go upstairs", vote_count=1)
        assert success is True
        assert votes[0]["actions"][0]["action"] == "UNKNOWN"

    @patch("vln_instruction_parser.llm._call_backend")
    def test_compact_draft_preserved(self, mock_call):
        """Raw votes should preserve compact draft fields."""
        actions = [
            _make_action("a1", "MOVE_FORWARD", "straight", [{"role": "path", "relation": "along", "landmark": "hallway"}]),
            _make_action("a2", "TURN", "left", [{"role": "where", "relation": "at", "landmark": "sofa"}]),
        ]
        mock_call.return_value = {
            "actions": actions,
            "order": [{"before": "a1", "after": "a2"}],
            "constraints": [],
            "excluded": [],
        }

        success, votes = parse_with_llm("Before turning left, go straight", vote_count=1)
        assert success is True
        assert len(votes[0]["order"]) == 1
        assert votes[0]["order"][0]["before"] == "a1"
