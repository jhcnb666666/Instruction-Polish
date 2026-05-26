"""End-to-end tests for the improved VLN parser pipeline."""

from unittest.mock import patch, MagicMock
import pytest
from vln_instruction_parser.parser import (
    parse_instruction_llm,
    parse_instruction_auto,
    apply_plan_confidence_policy,
)
from vln_instruction_parser.validator import validate_result
from vln_instruction_parser.schema import (
    ParseResult,
    Task,
    Feature,
    Constraint,
    AlternativePlan,
    BacktrackingResult,
    StepCandidateGroup,
    BacktrackingCandidate,
    result_to_dict,
    result_from_dict,
)
from vln_instruction_parser.semantic_compiler import compile_draft


def _plan(confidence, tasks=None, constraints=None):
    return {
        "status": "ok",
        "confidence": confidence,
        "tasks": tasks if tasks is not None else [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
        "constraints": constraints if constraints is not None else [],
    }


class TestVerifierIntegration:
    """Verifier reranker drives confidence policy decisions."""

    @patch("vln_instruction_parser.llm.verify_step_candidates")
    @patch("vln_instruction_parser.llm.verify_candidate_plans")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_verifier_c1_096_ok_no_backtracking(self, mock_parse, mock_verify, mock_step_verify):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ] * 3)
        mock_verify.return_value = [
            {"candidate_id": "p1", "confidence": 0.96},
        ]
        mock_step_verify.return_value = {
            "step_confidences": [{"step_id": 1, "confidence": 0.97}],
            "candidate_confidences": [],
        }
        result = parse_instruction_llm("Turn left.", fallback_to_rules=False, vote_count=3)
        assert result["status"] == "ok"
        assert result["confidence"] == 1.0
        assert result["alternatives"] == []
        assert result.get("backtracking", {}) == {"step_candidates": []}
        validate_result(result)

    @patch("vln_instruction_parser.llm.verify_step_candidates")
    @patch("vln_instruction_parser.llm.verify_candidate_plans")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_verifier_c1_095_c2_081_needs_review(self, mock_parse, mock_verify, mock_step_verify):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "TURN", "direction": "right", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ] + [
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ])
        mock_verify.return_value = [
            {"candidate_id": "p1", "confidence": 0.95},
            {"candidate_id": "p2", "confidence": 0.81},
        ]
        mock_step_verify.return_value = {
            "step_confidences": [{"step_id": 1, "confidence": 0.95}],
            "candidate_confidences": [{"step_id": 1, "rank": 2, "confidence": 0.81}],
        }
        result = parse_instruction_llm("Turn left.", fallback_to_rules=False, vote_count=3)
        # Close competitor at plan level => needs_review
        assert result["status"] == "needs_review"
        assert result["alternatives"] == []
        # Backtracking may contain the local candidate if step verifier passed
        validate_result(result)

    @patch("vln_instruction_parser.llm.verify_step_candidates")
    @patch("vln_instruction_parser.llm.verify_candidate_plans")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_verifier_c1_095_c2_080_ok(self, mock_parse, mock_verify, mock_step_verify):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "TURN", "direction": "right", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ] + [
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ])
        mock_verify.return_value = [
            {"candidate_id": "p1", "confidence": 0.95},
            {"candidate_id": "p2", "confidence": 0.80},
        ]
        mock_step_verify.return_value = {
            "step_confidences": [{"step_id": 1, "confidence": 0.95}],
            "candidate_confidences": [],
        }
        result = parse_instruction_llm("Turn left.", fallback_to_rules=False, vote_count=3)
        # 0.95 - 0.80 = 0.15, NOT strictly < 0.15, so ok
        assert result["status"] == "ok"
        assert result["alternatives"] == []
        validate_result(result)

    @patch("vln_instruction_parser.llm.verify_step_candidates")
    @patch("vln_instruction_parser.llm.verify_candidate_plans")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_verifier_c1_090_c2_071_needs_review(self, mock_parse, mock_verify, mock_step_verify):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "MOVE_FORWARD", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "STOP", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ])
        mock_verify.return_value = [
            {"candidate_id": "p1", "confidence": 0.90},
            {"candidate_id": "p2", "confidence": 0.71},
            {"candidate_id": "p3", "confidence": 0.69},
        ]
        mock_step_verify.return_value = {
            "step_confidences": [
                {"step_id": 1, "confidence": 0.92},
                {"step_id": 2, "confidence": 0.88},
                {"step_id": 3, "confidence": 0.90},
            ],
            "candidate_confidences": [
                {"step_id": 1, "rank": 2, "confidence": 0.71},
            ],
        }
        result = parse_instruction_llm("Go straight then turn left and stop.", fallback_to_rules=False, vote_count=3)
        assert result["status"] == "needs_review"
        assert result["alternatives"] == []
        validate_result(result)

    @patch("vln_instruction_parser.llm.verify_step_candidates")
    @patch("vln_instruction_parser.llm.verify_candidate_plans")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_verifier_c1_089_c2_075_c3_070_needs_review(self, mock_parse, mock_verify, mock_step_verify):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "MOVE_FORWARD", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "STOP", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ])
        mock_verify.return_value = [
            {"candidate_id": "p1", "confidence": 0.89},
            {"candidate_id": "p2", "confidence": 0.75},
            {"candidate_id": "p3", "confidence": 0.70},
        ]
        mock_step_verify.return_value = {
            "step_confidences": [
                {"step_id": 1, "confidence": 0.90},
                {"step_id": 2, "confidence": 0.88},
                {"step_id": 3, "confidence": 0.89},
            ],
            "candidate_confidences": [
                {"step_id": 1, "rank": 2, "confidence": 0.75},
                {"step_id": 1, "rank": 3, "confidence": 0.70},
            ],
        }
        result = parse_instruction_llm("Go straight then turn left and stop.", fallback_to_rules=False, vote_count=3)
        assert result["status"] == "needs_review"
        assert result["alternatives"] == []
        validate_result(result)

    @patch("vln_instruction_parser.llm.verify_step_candidates")
    @patch("vln_instruction_parser.llm.verify_candidate_plans")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_verifier_failure_degradation(self, mock_parse, mock_verify, mock_step_verify):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "TURN", "direction": "right", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ])
        mock_verify.return_value = None
        result = parse_instruction_llm("Turn left.", fallback_to_rules=False, vote_count=2)
        assert result["status"] == "needs_review"
        assert result["confidence"] == 0.0
        assert result.get("reason") == "confidence_verification_unavailable"
        assert result["alternatives"] == []
        validate_result(result)

    @patch("vln_instruction_parser.llm.verify_step_candidates")
    @patch("vln_instruction_parser.llm.verify_candidate_plans")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_backtracking_preserve_terminate_and_constraints(self, mock_parse, mock_verify, mock_step_verify):
        """Main plan must preserve full tasks and constraints."""
        mock_parse.return_value = (True, [
            {"actions": [
                {"id": "a1", "action": "MOVE_FORWARD", "features": [
                    {"role": "terminate", "landmark": "door", "trigger": "reach"}
                ]},
            ], "order": [], "constraints": [
                {"type": "forbidden_action", "action": "ENTER", "features": []}
            ], "excluded": []},
            {"actions": [
                {"id": "a1", "action": "TURN", "direction": "left", "features": [
                    {"role": "where", "relation": "at", "landmark": "door"}
                ]},
            ], "order": [], "constraints": [], "excluded": []},
        ])
        mock_verify.return_value = [
            {"candidate_id": "p1", "confidence": 0.92},
            {"candidate_id": "p2", "confidence": 0.83},
        ]
        mock_step_verify.return_value = {
            "step_confidences": [{"step_id": 1, "confidence": 0.92}],
            "candidate_confidences": [{"step_id": 1, "rank": 2, "confidence": 0.83}],
        }
        result = parse_instruction_llm("Go to the door.", fallback_to_rules=False, vote_count=2)
        assert result["status"] == "needs_review"
        # Main plan has terminate feature and constraint
        assert any(f.get("role") == "terminate" for f in result["tasks"][0].get("features", []))
        assert len(result["constraints"]) == 1
        # No full-plan alternatives anymore
        assert result["alternatives"] == []
        validate_result(result)

    @patch("vln_instruction_parser.llm.verify_step_candidates")
    @patch("vln_instruction_parser.llm.verify_candidate_plans")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_three_different_votes_kept(self, mock_parse, mock_verify, mock_step_verify):
        """All three votes different -> all three candidates enter verifier."""
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "MOVE_FORWARD", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "STOP", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ])
        mock_verify.return_value = [
            {"candidate_id": "p1", "confidence": 0.88},
            {"candidate_id": "p2", "confidence": 0.75},
            {"candidate_id": "p3", "confidence": 0.70},
        ]
        mock_step_verify.return_value = {
            "step_confidences": [{"step_id": 1, "confidence": 0.88}],
            "candidate_confidences": [
                {"step_id": 1, "rank": 2, "confidence": 0.75},
                {"step_id": 1, "rank": 3, "confidence": 0.70},
            ],
        }
        result = parse_instruction_llm("Do something.", fallback_to_rules=False, vote_count=3)
        assert result["status"] == "needs_review"
        assert result["alternatives"] == []
        validate_result(result)


class TestSemanticCompilerExcludedVertical:
    def test_excluded_stairs_then_turn_left_ok(self):
        """Instead of taking the stairs, turn left -> valid 2D TURN."""
        draft = {
            "actions": [
                {"id": "a1", "action": "MOVE_FORWARD", "features": [
                    {"role": "target", "relation": "toward", "landmark": "stairs"}
                ]},
                {"id": "a2", "action": "TURN", "direction": "left", "features": [
                    {"role": "where", "relation": "at", "landmark": "door"}
                ]},
            ],
            "order": [],
            "constraints": [],
            "excluded": ["a1"],
        }
        result = compile_draft(draft)
        assert result["status"] == "ok"
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["action"] == "TURN"

    def test_active_upstairs_then_turn_left_unsupported(self):
        """Go upstairs, then turn left -> unsupported."""
        draft = {
            "actions": [
                {"id": "a1", "action": "MOVE_FORWARD", "features": [
                    {"role": "target", "relation": "toward", "landmark": "upstairs"}
                ]},
                {"id": "a2", "action": "TURN", "direction": "left", "features": [
                    {"role": "where", "relation": "at", "landmark": "door"}
                ]},
            ],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        assert result["status"] == "unsupported"
        assert result["tasks"] == []

    def test_cyclic_order_returns_needs_review(self):
        """Circular dependency returns needs_review, not fallback order."""
        draft = {
            "actions": [
                {"id": "a1", "action": "TURN", "direction": "left", "features": []},
                {"id": "a2", "action": "MOVE_FORWARD", "features": []},
            ],
            "order": [
                {"before": "a1", "after": "a2"},
                {"before": "a2", "after": "a1"},
            ],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        assert result["status"] == "needs_review"
        assert result.get("reason") == "cyclic_execution_order"


class TestSchemaRoundtrip:
    def test_result_with_backtracking_roundtrip(self):
        result = ParseResult(
            status="needs_review",
            confidence=0.93,
            tasks=[
                Task(
                    step_id=1,
                    action="MOVE_FORWARD",
                    features=[Feature(role="terminate", trigger="reach", landmark="door")],
                    confidence=0.91,
                )
            ],
            constraints=[
                Constraint(type="forbidden_action", action="ENTER", features=[])
            ],
            alternatives=[],
            backtracking=BacktrackingResult(
                step_candidates=[
                    StepCandidateGroup(
                        step_id=1,
                        candidates=[
                            BacktrackingCandidate(
                                rank=2,
                                step_id=1,
                                action="TURN",
                                direction="left",
                                features=[],
                                confidence=0.82,
                            )
                        ],
                    )
                ]
            ),
        )
        d = result_to_dict(result)
        validate_result(d)
        back = result_from_dict(d)
        assert back.status == "needs_review"
        assert back.confidence == 0.93
        assert len(back.tasks) == 1
        assert len(back.constraints) == 1
        assert len(back.alternatives) == 0
        assert len(back.backtracking.step_candidates) == 1
        assert back.backtracking.step_candidates[0].candidates[0].rank == 2


class TestNoInternalMetadataInOutput:
    @patch("vln_instruction_parser.llm.verify_step_candidates")
    @patch("vln_instruction_parser.llm.verify_candidate_plans")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_output_has_no_raw_text_or_candidate_id(self, mock_parse, mock_verify, mock_step_verify):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ] * 3)
        mock_verify.return_value = [
            {"candidate_id": "p1", "confidence": 0.96},
        ]
        mock_step_verify.return_value = {
            "step_confidences": [{"step_id": 1, "confidence": 0.97}],
            "candidate_confidences": [],
        }
        result = parse_instruction_llm("Turn left.", fallback_to_rules=False, vote_count=3)
        # Forbidden keys must not appear
        forbidden = {"raw_text", "original_instruction", "canonical_instruction",
                     "source_order", "candidate_id", "vote_support"}
        result_keys = set(result.keys())
        assert not (forbidden & result_keys)
        for task in result.get("tasks", []):
            assert not (forbidden & set(task.keys()))
        for group in result.get("backtracking", {}).get("step_candidates", []):
            assert not (forbidden & set(group.keys()))
            for cand in group.get("candidates", []):
                assert not (forbidden & set(cand.keys()))
        validate_result(result)


class TestRulePathConfidence:
    def test_rule_path_not_auto_accepted(self):
        result = parse_instruction_auto("Turn left.")
        # Even simple rule path no longer gets unconditional 1.0
        # But since auto now routes through LLM, this actually calls LLM.
        # We test parse_instruction directly instead.
        from vln_instruction_parser.parser import parse_instruction
        result = parse_instruction("Turn left.")
        assert result["status"] == "ok"
        assert result["confidence"] == 0.85
        assert result["alternatives"] == []
