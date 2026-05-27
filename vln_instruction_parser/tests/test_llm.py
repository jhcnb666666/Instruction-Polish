"""Tests for LLM provider module with mocked backend — compact semantic drafts."""

import json
from unittest.mock import MagicMock, patch

import pytest
from vln_instruction_parser.llm import (
    audit_plans_against_instruction,
    parse_with_llm,
    split_instruction_semantically,
)


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
    def test_failed_vote_does_not_discard_successful_votes(self, mock_call):
        valid = {
            "actions": [_make_action("a1", "TURN", "left")],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        mock_call.side_effect = [valid, None, valid]

        success, votes = parse_with_llm("Turn left.", vote_count=3)

        assert success is True
        assert len(votes) == 2
        assert mock_call.call_count == 3

    @patch("vln_instruction_parser.llm._call_backend")
    def test_malformed_action_vote_is_discarded(self, mock_call):
        mock_call.side_effect = [
            {"actions": ["not-an-action"], "order": [], "constraints": [], "excluded": []},
            {"actions": [_make_action("a1", "STOP")], "order": [], "constraints": [], "excluded": []},
        ]

        success, votes = parse_with_llm("Stop.", vote_count=2)

        assert success is True
        assert len(votes) == 1

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


class TestAuditDiagnostics:
    @patch("vln_instruction_parser.llm._call_local")
    def test_invalid_step_id_records_mapping_failure(self, mock_call, caplog):
        mock_call.return_value = {
            "audits": [{
                "candidate_id": "p1",
                "plan_confidence": 0.8,
                "blocking_issues": [],
                "step_confidences": [{"step_id": 99, "confidence": 0.8}],
            }]
        }
        plan = {
            "candidate_id": "p1",
            "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
            "constraints": [],
        }

        with patch("vln_instruction_parser.llm._get_config", return_value={
            "backend": "local",
            "model_path": "mock-model",
            "max_tokens": 32,
        }):
            with caplog.at_level("WARNING", logger="vln_instruction_parser.llm"):
                result = audit_plans_against_instruction("Turn left.", [plan])

        assert result is None
        assert "audit_step_mapping_invalid:step_id" in caplog.text


class TestSemanticSegmentation:
    @patch("vln_instruction_parser.llm._call_local")
    def test_local_segmenter_returns_contiguous_excerpts(self, mock_call):
        mock_call.return_value = {"segments": ["Walk forward. Turn left.", "Stop."]}
        with patch("vln_instruction_parser.llm._get_config", return_value={
            "backend": "local",
            "model_path": "mock-model",
            "max_tokens": 64,
        }):
            result = split_instruction_semantically(
                "Walk forward. Turn left. Stop.",
                max_sentences=2,
                max_phases=3,
                max_words=80,
            )
        assert result == ["Walk forward. Turn left.", "Stop."]
        assert mock_call.call_args.kwargs["temperature"] == 0.0
        assert "3 recognizable navigation action phases" in mock_call.call_args.kwargs["user_prompt"]
        assert "80 English words" in mock_call.call_args.kwargs["user_prompt"]

    @patch("vln_instruction_parser.llm._call_local", return_value={"segments": ["", "Stop."]})
    def test_segmenter_rejects_empty_segment(self, _mock_call):
        with patch("vln_instruction_parser.llm._get_config", return_value={
            "backend": "local",
            "model_path": "mock-model",
            "max_tokens": 64,
        }):
            assert split_instruction_semantically("Walk forward. Stop.") is None
