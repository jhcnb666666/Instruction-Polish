"""Tests for multi-sentence VLN parsing and merge logic."""

import os
from unittest.mock import patch, call

from vln_instruction_parser.segmenter import (
    split_sentences_for_llm,
    has_cross_sentence_dependency,
)
from vln_instruction_parser.parser import (
    parse_instruction_auto,
    _merge_sentence_results,
    _merge_initial_contributions,
    _merge_segment_results_into_plans,
    _valid_semantic_segments,
    _generate_initial_contribution_for_sentence,
    _run_global_postprocess,
    _sentence_split_threshold,
    _segment_max_phases,
    _segment_max_words,
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

    def test_backtracking_merge(self):
        r1 = {
            "status": "ok", "confidence": 0.95,
            "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}],
            "constraints": [],
            "backtracking": {
                "step_candidates": [
                    {
                        "step_id": 1,
                        "candidates": [
                            {"rank": 2, "step_id": 1, "action": "TURN", "direction": "right", "features": [], "confidence": 0.85}
                        ],
                    }
                ]
            },
        }
        r2 = {
            "status": "ok", "confidence": 0.93,
            "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
            "constraints": [],
            "backtracking": {"step_candidates": []},
        }
        merged = _merge_sentence_results([r1, r2], "Walk forward. Turn left.")
        bt = merged.get("backtracking", {})
        assert len(bt.get("step_candidates", [])) == 1
        group = bt["step_candidates"][0]
        assert group["step_id"] == 1
        assert len(group["candidates"]) == 1
        cand = group["candidates"][0]
        assert cand["rank"] == 2
        assert cand["action"] == "TURN"
        assert cand["direction"] == "right"

    def test_empty_results(self):
        merged = _merge_sentence_results([], "")
        assert merged["status"] == "none"
        assert merged["reason"] == "no_action"


class TestMergeInitialContributions:
    def test_renumbers_steps_across_sentences(self):
        c1 = {"tasks": [{"step_id": 1, "action": "MOVE_FORWARD"}], "constraints": []}
        c2 = {"tasks": [{"step_id": 1, "action": "TURN"}, {"step_id": 2, "action": "STOP"}], "constraints": []}
        c3 = {"tasks": [{"step_id": 1, "action": "MOVE_FORWARD"}], "constraints": []}
        merged = _merge_initial_contributions([c1, c2, c3])
        assert merged["candidate_id"] == "merged"
        assert len(merged["tasks"]) == 4
        assert [t["step_id"] for t in merged["tasks"]] == [1, 2, 3, 4]
        assert merged["tasks"][0]["action"] == "MOVE_FORWARD"
        assert merged["tasks"][1]["action"] == "TURN"
        assert merged["tasks"][2]["action"] == "STOP"
        assert merged["tasks"][3]["action"] == "MOVE_FORWARD"

    def test_dedup_constraints(self):
        c1 = {
            "tasks": [],
            "constraints": [{"type": "forbidden_action", "action": "ENTER", "features": []}],
        }
        c2 = {
            "tasks": [],
            "constraints": [{"type": "forbidden_action", "action": "ENTER", "features": []}],
        }
        merged = _merge_initial_contributions([c1, c2])
        assert len(merged["constraints"]) == 1

    def test_constraints_only_with_tasks(self):
        c1 = {
            "tasks": [],
            "constraints": [{"type": "forbidden_action", "action": "ENTER", "features": []}],
        }
        c2 = {
            "tasks": [{"step_id": 1, "action": "TURN"}],
            "constraints": [],
        }
        merged = _merge_initial_contributions([c1, c2])
        assert len(merged["tasks"]) == 1
        assert merged["tasks"][0]["step_id"] == 1
        assert len(merged["constraints"]) == 1

    def test_empty_contributions(self):
        merged = _merge_initial_contributions([])
        assert merged["tasks"] == []
        assert merged["constraints"] == []


class TestSentenceSplitThreshold:
    def setup_method(self):
        # Clean env vars before each test
        for key in (
            "VLN_SENTENCE_SPLIT_THRESHOLD",
            "VLN_MAX_SENTENCE_CHUNKS",
            "VLN_SEGMENT_MAX_PHASES",
            "VLN_SEGMENT_MAX_WORDS",
        ):
            os.environ.pop(key, None)

    def teardown_method(self):
        for key in (
            "VLN_SENTENCE_SPLIT_THRESHOLD",
            "VLN_MAX_SENTENCE_CHUNKS",
            "VLN_SEGMENT_MAX_PHASES",
            "VLN_SEGMENT_MAX_WORDS",
        ):
            os.environ.pop(key, None)

    def test_default_is_five(self):
        assert _sentence_split_threshold() == 5

    def test_new_env_var(self):
        os.environ["VLN_SENTENCE_SPLIT_THRESHOLD"] = "5"
        assert _sentence_split_threshold() == 5

    def test_old_env_var_alias(self):
        os.environ["VLN_MAX_SENTENCE_CHUNKS"] = "3"
        assert _sentence_split_threshold() == 3

    def test_sentence_limit_is_capped_at_five(self):
        os.environ["VLN_SENTENCE_SPLIT_THRESHOLD"] = "10"
        os.environ["VLN_MAX_SENTENCE_CHUNKS"] = "3"
        assert _sentence_split_threshold() == 5

    def test_phase_and_word_defaults(self):
        assert _segment_max_phases() == 5
        assert _segment_max_words() == 120

    def test_stricter_phase_and_word_limits_take_effect(self):
        os.environ["VLN_SEGMENT_MAX_PHASES"] = "3"
        os.environ["VLN_SEGMENT_MAX_WORDS"] = "80"
        assert _segment_max_phases() == 3
        assert _segment_max_words() == 80

    def test_phase_and_word_limits_cannot_relax_hard_budget(self):
        os.environ["VLN_SEGMENT_MAX_PHASES"] = "20"
        os.environ["VLN_SEGMENT_MAX_WORDS"] = "500"
        assert _segment_max_phases() == 5
        assert _segment_max_words() == 120


class TestParseInstructionAutoLongPath:
    @patch("vln_instruction_parser.parser._run_global_postprocess")
    @patch("vln_instruction_parser.llm.split_instruction_semantically")
    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_short_path_routes_to_whole_parse(self, mock_llm, mock_split, mock_post):
        """Instructions of at most five sentences remain one semantic parse."""
        mock_llm.return_value = {
            "status": "ok",
            "confidence": 0.95,
            "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}],
            "constraints": [],
            "backtracking": {"step_candidates": []},
        }
        result = parse_instruction_auto("Walk forward. Turn left. Stop.")
        mock_llm.assert_called_once()
        assert mock_llm.call_args.kwargs["vote_count"] == 1
        mock_split.assert_not_called()
        mock_post.assert_not_called()
        assert result["status"] == "ok"

    @patch("vln_instruction_parser.parser._run_global_postprocess")
    @patch("vln_instruction_parser.llm.split_instruction_semantically")
    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_long_path_routes_semantic_segments(self, mock_llm, mock_split, mock_post):
        """Six sentences are grouped semantically and each group gets the full pipeline."""
        sentences = [f"Walk to point {i}." for i in range(1, 7)]
        text = " ".join(sentences)
        mock_split.return_value = [" ".join(sentences[:3]), " ".join(sentences[3:])]
        mock_llm.return_value = {
            "status": "ok", "confidence": 0.94,
            "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}],
            "constraints": [], "backtracking": {"step_candidates": []},
        }
        mock_post.return_value = {
            "status": "ok",
            "confidence": 0.92,
            "tasks": [{"step_id": i, "action": "MOVE_FORWARD", "features": []} for i in range(1, 3)],
            "constraints": [],
            "backtracking": {"step_candidates": []},
        }
        result = parse_instruction_auto(text)
        mock_split.assert_called_once()
        assert mock_split.call_args.kwargs["max_sentences"] == 5
        assert mock_split.call_args.kwargs["max_phases"] == 5
        assert mock_split.call_args.kwargs["max_words"] == 120
        assert mock_llm.call_count == 2
        assert all(call.kwargs["fallback_to_rules"] for call in mock_llm.call_args_list)
        mock_post.assert_called_once()
        assert result["status"] == "ok"

    @patch("vln_instruction_parser.parser._run_global_postprocess")
    @patch("vln_instruction_parser.llm.split_instruction_semantically")
    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_long_path_lifts_local_step_candidate(self, mock_llm, mock_split, mock_post):
        sentences = [f"Sentence {i}." for i in range(1, 7)]
        text = " ".join(sentences)
        mock_split.return_value = [" ".join(sentences[:3]), " ".join(sentences[3:])]
        mock_llm.side_effect = [
            {
                "status": "ok", "confidence": 0.92,
                "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
                "constraints": [],
                "backtracking": {"step_candidates": [{
                    "step_id": 1,
                    "candidates": [{"rank": 2, "step_id": 1, "action": "TURN", "direction": "right", "features": [], "confidence": 0.87}],
                }]},
            },
            {
                "status": "ok", "confidence": 0.95,
                "tasks": [{"step_id": 1, "action": "STOP", "features": []}],
                "constraints": [], "backtracking": {"step_candidates": []},
            },
        ]
        mock_post.return_value = {
            "status": "ok", "confidence": 0.92, "tasks": [],
            "constraints": [], "backtracking": {"step_candidates": []},
        }
        parse_instruction_auto(text)
        plans = mock_post.call_args.args[1]
        assert len(plans) == 2
        assert plans[0]["candidate_id"] == "merged_long"
        assert [t["step_id"] for t in plans[0]["tasks"]] == [1, 2]
        assert plans[1]["tasks"][0]["direction"] == "right"
        assert plans[1]["tasks"][1]["action"] == "STOP"

    @patch("vln_instruction_parser.parser._run_global_postprocess")
    @patch("vln_instruction_parser.llm.split_instruction_semantically")
    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_invalid_segmentation_falls_back_to_whole_parse(self, mock_llm, mock_split, mock_post):
        sentences = [f"Walk to point {i}." for i in range(1, 7)]
        text = " ".join(sentences)
        mock_split.return_value = ["rewritten text", "not the original input"]
        mock_llm.return_value = {
            "status": "ok", "confidence": 0.93,
            "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}],
            "constraints": [], "backtracking": {"step_candidates": []},
        }
        result = parse_instruction_auto(text)
        assert result["status"] == "ok"
        mock_llm.assert_called_once_with(
            text, fallback_to_rules=True, vote_count=1,
            base_url=None, model=None, temperature=None,
        )
        mock_post.assert_not_called()

    @patch("vln_instruction_parser.parser._run_global_postprocess")
    @patch("vln_instruction_parser.llm.split_instruction_semantically")
    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_unparsed_semantic_segment_falls_back_to_whole_parse(self, mock_llm, mock_split, mock_post):
        sentences = [f"Walk to point {i}." for i in range(1, 7)]
        text = " ".join(sentences)
        mock_split.return_value = [" ".join(sentences[:3]), " ".join(sentences[3:])]
        mock_llm.side_effect = [
            {"status": "ok", "confidence": 0.9, "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}], "constraints": [], "backtracking": {"step_candidates": []}},
            {"status": "needs_review", "confidence": 0.0, "tasks": [], "constraints": [], "backtracking": {"step_candidates": []}},
            {"status": "ok", "confidence": 0.91, "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}], "constraints": [], "backtracking": {"step_candidates": []}},
        ]
        result = parse_instruction_auto(text)
        assert result["status"] == "ok"
        assert mock_llm.call_args_list[-1].args[0] == text
        mock_post.assert_not_called()

    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_single_sentence_within_budget_uses_short_path(self, mock_llm):
        """A descriptive single sentence stays whole while it remains within budgets."""
        mock_llm.return_value = {
            "status": "ok",
            "confidence": 0.90,
            "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}],
            "constraints": [],
            "backtracking": {"step_candidates": []},
        }
        text = "Walk straight down the long hallway and pass the first two intersections keeping moving forward until you see a large blue painting on your right"
        result = parse_instruction_auto(text)
        mock_llm.assert_called_once()
        assert mock_llm.call_args.kwargs["vote_count"] == 1
        assert result["status"] == "ok"

    @patch("vln_instruction_parser.parser._run_global_postprocess")
    @patch("vln_instruction_parser.llm.split_instruction_semantically")
    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_single_sentence_with_six_action_phases_is_semantically_split(
        self, mock_llm, mock_split, mock_post
    ):
        text = "Walk forward, turn left, move forward, turn right, pass the desk, then stop."
        mock_split.return_value = [
            "Walk forward, turn left, move forward,",
            "turn right, pass the desk, then stop.",
        ]
        mock_llm.return_value = {
            "status": "ok", "confidence": 0.94,
            "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}],
            "constraints": [], "backtracking": {"step_candidates": []},
        }
        mock_post.return_value = {
            "status": "ok", "confidence": 0.93, "tasks": [],
            "constraints": [], "backtracking": {"step_candidates": []},
        }

        result = parse_instruction_auto(text)

        mock_split.assert_called_once()
        assert mock_llm.call_count == 2
        mock_post.assert_called_once()
        assert result["status"] == "ok"

    @patch("vln_instruction_parser.parser._run_global_postprocess")
    @patch("vln_instruction_parser.llm.split_instruction_semantically")
    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_single_sentence_over_word_budget_is_semantically_split(
        self, mock_llm, mock_split, mock_post
    ):
        words = ["Walk"] + ["forward"] * 120 + ["stop"]
        text = " ".join(words)
        mock_split.return_value = [" ".join(words[:61]), " ".join(words[61:])]
        mock_llm.return_value = {
            "status": "ok", "confidence": 0.94,
            "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}],
            "constraints": [], "backtracking": {"step_candidates": []},
        }
        mock_post.return_value = {
            "status": "ok", "confidence": 0.93, "tasks": [],
            "constraints": [], "backtracking": {"step_candidates": []},
        }

        result = parse_instruction_auto(text)

        mock_split.assert_called_once()
        assert mock_llm.call_count == 2
        assert result["status"] == "ok"

    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_short_path_respects_explicit_vote_count(self, mock_llm):
        mock_llm.return_value = {
            "status": "ok",
            "confidence": 0.95,
            "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
            "constraints": [],
            "backtracking": {"step_candidates": []},
        }
        parse_instruction_auto("Turn left.", vote_count=3)
        assert mock_llm.call_args.kwargs["vote_count"] == 3

    @patch("vln_instruction_parser.parser._run_global_postprocess")
    @patch("vln_instruction_parser.llm.split_instruction_semantically")
    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_semantic_segments_respect_explicit_vote_count(self, mock_llm, mock_split, mock_post):
        sentences = [f"Walk to point {i}." for i in range(1, 7)]
        text = " ".join(sentences)
        mock_split.return_value = [" ".join(sentences[:3]), " ".join(sentences[3:])]
        mock_llm.return_value = {"status": "ok", "confidence": 0.93, "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}], "constraints": [], "backtracking": {"step_candidates": []}}
        mock_post.return_value = {"status": "ok", "confidence": 0.92, "tasks": [], "constraints": [], "backtracking": {"step_candidates": []}}
        parse_instruction_auto(text, vote_count=3)
        assert mock_llm.call_count == 2
        assert all(call.kwargs["vote_count"] == 3 for call in mock_llm.call_args_list)

    @patch("vln_instruction_parser.llm.split_instruction_semantically")
    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_long_path_vertical_motion_is_rejected_before_segmentation(self, mock_llm, mock_split):
        text = "Walk forward. Turn left. Stop. Wait. Go upstairs. Turn right."
        result = parse_instruction_auto(text)
        assert result["status"] == "unsupported"
        assert result["reason"] == "vertical_motion_not_supported"
        mock_split.assert_not_called()
        mock_llm.assert_not_called()

    @patch("vln_instruction_parser.parser._run_global_postprocess")
    @patch("vln_instruction_parser.llm.split_instruction_semantically")
    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_global_fidelity_blocker_reparses_complete_instruction(
        self, mock_llm, mock_split, mock_post
    ):
        sentences = [f"Walk to point {i}." for i in range(1, 7)]
        text = " ".join(sentences)
        mock_split.return_value = [" ".join(sentences[:3]), " ".join(sentences[3:])]
        local_result = {
            "status": "ok", "confidence": 0.93,
            "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}],
            "constraints": [], "backtracking": {"step_candidates": []},
        }
        whole_result = {
            "status": "ok", "confidence": 0.97,
            "tasks": [{"step_id": 1, "action": "MOVE_FORWARD", "features": []}],
            "constraints": [], "backtracking": {"step_candidates": []},
        }
        mock_llm.side_effect = [local_result, local_result, whole_result]
        mock_post.return_value = {
            "status": "needs_review", "confidence": 0.71, "tasks": [],
            "constraints": [], "backtracking": {"step_candidates": []},
            "reason": "wrong_order",
        }

        result = parse_instruction_auto(text)

        assert result == whole_result
        assert mock_llm.call_count == 3
        assert mock_llm.call_args_list[-1].args[0] == text


class TestSemanticSegmentHelpers:
    def test_valid_segments_require_original_order_and_five_sentence_limit(self):
        sentences = [f"Move {i}." for i in range(1, 7)]
        text = " ".join(sentences)
        assert _valid_semantic_segments(text, [" ".join(sentences[:3]), " ".join(sentences[3:])], 5)
        assert not _valid_semantic_segments(text, [" ".join(sentences[3:]), " ".join(sentences[:3])], 5)
        assert not _valid_semantic_segments(text, [text], 5)

    def test_valid_segments_enforce_phase_and_word_budgets(self):
        phased = "Walk forward, turn left, stop, move forward."
        assert not _valid_semantic_segments(
            phased,
            ["Walk forward, turn left, stop,", "move forward."],
            max_sentences=5,
            max_phases=2,
            max_words=120,
        )

        words = ["Walk"] + ["quiet"] * 12
        text = " ".join(words)
        assert not _valid_semantic_segments(
            text,
            [" ".join(words[:9]), " ".join(words[9:])],
            max_sentences=5,
            max_phases=5,
            max_words=8,
        )

    def test_merge_segment_results_preserves_constraints_and_lifts_candidates(self):
        plans = _merge_segment_results_into_plans([
            {
                "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
                "constraints": [{"type": "forbidden_action", "action": "ENTER", "features": []}],
                "backtracking": {"step_candidates": [{
                    "step_id": 1,
                    "candidates": [{"rank": 2, "step_id": 1, "action": "TURN", "direction": "right", "features": [], "confidence": 0.8}],
                }]},
            },
            {
                "tasks": [{"step_id": 1, "action": "STOP", "features": []}],
                "constraints": [{"type": "forbidden_action", "action": "ENTER", "features": []}],
                "backtracking": {"step_candidates": []},
            },
        ])
        assert len(plans) == 2
        assert len(plans[0]["constraints"]) == 1
        assert plans[1]["tasks"][0]["direction"] == "right"
        assert plans[1]["tasks"][1]["step_id"] == 2


class TestParseInstructionAutoMultiSentence:
    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_whole_instruction_routed_to_llm(self, mock_llm):
        """All instructions (single or multi-sentence) route as a whole to fidelity audit."""
        mock_llm.return_value = {
            "status": "ok",
            "confidence": 0.94,
            "tasks": [
                {"step_id": 1, "action": "MOVE_FORWARD", "features": [], "confidence": 0.94}
            ],
            "constraints": [],
            "backtracking": {"step_candidates": []},
        }
        result = parse_instruction_auto("Walk to the door. Then turn when you see it.")
        mock_llm.assert_called_once()
        assert result["status"] == "ok"
        assert result["confidence"] == 0.94

    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_multi_sentence_whole_parse(self, mock_llm):
        """Multi-sentence instructions are parsed as a whole, not split per sentence."""
        mock_llm.return_value = {
            "status": "ok",
            "confidence": 0.92,
            "tasks": [
                {"step_id": 1, "action": "MOVE_FORWARD", "features": [], "confidence": 0.92},
                {"step_id": 2, "action": "TURN", "direction": "left", "features": [], "confidence": 0.90},
            ],
            "constraints": [],
            "backtracking": {"step_candidates": []},
        }
        result = parse_instruction_auto("Walk forward. Turn left.")
        mock_llm.assert_called_once()
        assert result["status"] == "ok"
        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["step_id"] == 1
        assert result["tasks"][1]["step_id"] == 2

    @patch("vln_instruction_parser.parser.parse_instruction_llm")
    def test_single_sentence_routes_directly(self, mock_llm):
        """Single sentence should route to parse_instruction_llm directly."""
        mock_llm.return_value = {
            "status": "ok",
            "confidence": 0.96,
            "tasks": [],
            "constraints": [],
            "alternatives": [],
            "backtracking": {"step_candidates": []},
        }
        result = parse_instruction_auto("Turn left.")
        mock_llm.assert_called_once()
        assert result["status"] == "ok"

    def test_preflight_empty(self):
        result = parse_instruction_auto("")
        assert result["status"] == "ok"
        assert result["tasks"] == []

    def test_rule_path_vertical_motion(self):
        """Rule path still rejects active vertical motion preflight."""
        from vln_instruction_parser.parser import parse_instruction
        result = parse_instruction("Go upstairs and turn left.")
        assert result["status"] == "unsupported"
        assert result["reason"] == "vertical_motion_not_supported"
