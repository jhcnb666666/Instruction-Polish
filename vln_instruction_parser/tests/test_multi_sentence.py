"""Tests for multi-sentence VLN parsing and merge logic."""

from unittest.mock import patch

from vln_instruction_parser.segmenter import (
    split_sentences_for_llm,
    has_cross_sentence_dependency,
)
from vln_instruction_parser.parser import (
    parse_instruction_auto,
    _merge_sentence_results,
)


class TestSplitSentencesForLLM:
    def test_basic_split(self):
        text = "Walk to the table. Turn right. Stop."
        sentences = split_sentences_for_llm(text)
        assert sentences == ["Walk to the table.", "Turn right.", "Stop."]

    def test_question_exclamation(self):
        text = "Go straight? Turn left! Stop."
        sentences = split_sentences_for_llm(text)
        assert sentences == ["Go straight?", "Turn left!", "Stop."]

    def test_abbreviations_protected(self):
        text = "Go past Dr. Smith's office, e.g. the room near the stairs. Then turn left."
        sentences = split_sentences_for_llm(text)
        # "Dr." and "e.g." should NOT split sentences
        assert len(sentences) == 2
        assert "Dr. Smith's office" in sentences[0]
        assert "e.g." in sentences[0]
        assert "turn left" in sentences[1]

    def test_decimal_protected(self):
        text = "Walk 3.14 meters. Then stop."
        sentences = split_sentences_for_llm(text)
        assert len(sentences) == 2
        assert "3.14 meters" in sentences[0]

    def test_empty(self):
        assert split_sentences_for_llm("") == []

    def test_single_sentence_no_period(self):
        assert split_sentences_for_llm("Turn left") == ["Turn left"]


class TestHasCrossSentenceDependency:
    def test_no_dependency(self):
        sentences = ["Walk to the door.", "Turn left.", "Stop."]
        has_dep, reason = has_cross_sentence_dependency(sentences)
        assert not has_dep
        assert reason == ""

    def test_pronoun_it(self):
        sentences = ["Walk to the door.", "Turn when you see it."]
        has_dep, reason = has_cross_sentence_dependency(sentences)
        assert has_dep
        assert "backref 'it'" in reason

    def test_pronoun_this(self):
        sentences = ["Enter the room.", "This should be your destination."]
        has_dep, reason = has_cross_sentence_dependency(sentences)
        assert has_dep
        assert "backref 'this'" in reason

    def test_do_so(self):
        sentences = ["Walk to the door.", "Then do so carefully."]
        has_dep, reason = has_cross_sentence_dependency(sentences)
        assert has_dep
        assert "backref 'do so'" in reason

    def test_instead_at_start(self):
        sentences = ["Walk to the door.", "Instead, turn left."]
        has_dep, reason = has_cross_sentence_dependency(sentences)
        assert has_dep
        assert "rewrite starter 'instead'" in reason

    def test_single_sentence_safe(self):
        sentences = ["Walk to the door and turn left."]
        has_dep, reason = has_cross_sentence_dependency(sentences)
        assert not has_dep

    def test_safe_continuer_then(self):
        sentences = ["Walk to the door.", "Then turn left."]
        has_dep, reason = has_cross_sentence_dependency(sentences)
        assert not has_dep


class TestMergeSentenceResults:
    def test_basic_merge(self):
        r1 = {
            "status": "ok",
            "confidence": 0.95,
            "tasks": [
                {"step_id": 1, "action": "MOVE_FORWARD", "features": [], "confidence": 0.95}
            ],
            "constraints": [],
            "alternatives": [],
        }
        r2 = {
            "status": "ok",
            "confidence": 0.93,
            "tasks": [
                {"step_id": 1, "action": "TURN", "direction": "left", "features": [], "confidence": 0.93}
            ],
            "constraints": [],
            "alternatives": [],
        }
        merged = _merge_sentence_results([r1, r2], "Walk forward. Turn left.")
        assert merged["status"] == "ok"
        assert len(merged["tasks"]) == 2
        assert merged["tasks"][0]["step_id"] == 1
        assert merged["tasks"][1]["step_id"] == 2
        assert merged["confidence"] == 0.93

    def test_status_propagation_unsupported(self):
        r1 = {"status": "ok", "confidence": 1.0, "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}], "constraints": [], "alternatives": []}
        r2 = {"status": "unsupported", "confidence": 1.0, "tasks": [], "constraints": [], "alternatives": [], "reason": "vertical_motion_not_supported"}
        merged = _merge_sentence_results([r1, r2], "Walk forward. Go upstairs.")
        assert merged["status"] == "unsupported"

    def test_status_propagation_needs_review(self):
        r1 = {"status": "ok", "confidence": 1.0, "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}], "constraints": [], "alternatives": []}
        r2 = {"status": "needs_review", "confidence": 0.0, "tasks": [], "constraints": [], "alternatives": [], "reason": "sentence_chunk_unparsed"}
        merged = _merge_sentence_results([r1, r2], "Walk forward. ???")
        assert merged["status"] == "needs_review"
        assert "sentence_chunk_unparsed" in merged.get("reason", "")

    def test_constraint_union(self):
        r1 = {
            "status": "ok", "confidence": 1.0,
            "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}],
            "constraints": [{"type": "forbidden_action", "action": "ENTER", "features": []}],
            "alternatives": [],
        }
        r2 = {
            "status": "ok", "confidence": 1.0,
            "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
            "constraints": [{"type": "forbidden_action", "action": "ENTER", "features": []}],
            "alternatives": [],
        }
        merged = _merge_sentence_results([r1, r2], "Walk forward. Turn left.")
        assert len(merged["constraints"]) == 1  # deduped

    def test_alternative_generation(self):
        r1 = {
            "status": "ok", "confidence": 0.95,
            "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}],
            "constraints": [],
            "alternatives": [
                {"rank": 2, "confidence": 0.85, "tasks": [{"step_id": 1, "action": "TURN", "direction": "right", "features": []}], "constraints": []}
            ],
        }
        r2 = {
            "status": "ok", "confidence": 0.93,
            "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
            "constraints": [],
            "alternatives": [],
        }
        merged = _merge_sentence_results([r1, r2], "Walk forward. Turn left.")
        assert len(merged["alternatives"]) == 1
        alt = merged["alternatives"][0]
        assert alt["rank"] == 2
        assert len(alt["tasks"]) == 2
        assert alt["tasks"][0]["action"] == "TURN"
        assert alt["tasks"][0]["direction"] == "right"
        assert alt["tasks"][1]["action"] == "TURN"
        assert alt["tasks"][1]["direction"] == "left"

    def test_empty_results(self):
        merged = _merge_sentence_results([], "")
        assert merged["status"] == "none"
        assert merged["reason"] == "no_action"


class TestParseInstructionAutoMultiSentence:
    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_cross_sentence_dependency_routes_whole(self, mock_llm):
        """If cross-sentence dependency detected, parse whole instruction but mark needs_review."""
        mock_llm.return_value = {
            "status": "ok",
            "confidence": 1.0,
            "tasks": [
                {"step_id": 1, "action": "MOVE_FORWARD", "features": [], "confidence": 1.0}
            ],
            "constraints": [],
            "alternatives": [],
        }
        result = parse_instruction_auto("Walk to the door. Then turn when you see it.")
        mock_llm.assert_called_once()
        assert result["status"] == "needs_review"
        assert result["reason"] == "cross_sentence_dependency_requires_review"
        assert result["confidence"] <= 0.9

    @patch("vln_instruction_parser.parser._parse_single_sentence_llm")
    def test_multi_sentence_parses_each(self, mock_single):
        """Multi-sentence without dependencies parses each sentence independently."""
        mock_single.side_effect = [
            {
                "status": "ok",
                "confidence": 0.95,
                "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": [], "confidence": 0.95}],
                "constraints": [],
                "alternatives": [],
            },
            {
                "status": "ok",
                "confidence": 0.93,
                "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": [], "confidence": 0.93}],
                "constraints": [],
                "alternatives": [],
            },
        ]
        result = parse_instruction_auto("Walk forward. Turn left.")
        assert mock_single.call_count == 2
        assert result["status"] == "ok"
        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["step_id"] == 1
        assert result["tasks"][1]["step_id"] == 2

    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_single_sentence_routes_directly(self, mock_llm):
        """Single sentence should route to parse_instruction_llm directly."""
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

    def test_preflight_empty(self):
        result = parse_instruction_auto("")
        assert result["status"] == "ok"
        assert result["tasks"] == []

    def test_preflight_vertical_motion(self):
        result = parse_instruction_auto("Go upstairs and turn left.")
        assert result["status"] == "unsupported"
        assert result["reason"] == "vertical_motion_not_supported"
