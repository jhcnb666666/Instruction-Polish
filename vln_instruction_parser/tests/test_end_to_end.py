"""End-to-end tests for the improved VLN parser pipeline."""

from unittest.mock import patch, MagicMock
import pytest
from vln_instruction_parser.parser import (
    parse_instruction_llm,
    parse_instruction_auto,
    _run_global_postprocess,
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


def _audit(cid, plan_conf, step_confs=None, blocking=None):
    return {
        "candidate_id": cid,
        "plan_confidence": plan_conf,
        "blocking_issues": list(blocking or []),
        "step_confidences": [
            {"step_id": sid, "confidence": c}
            for sid, c in (step_confs or {}).items()
        ],
    }


class TestFidelityAuditIntegration:
    """Fidelity audit drives confidence and status decisions."""

    @patch("vln_instruction_parser.llm.generate_step_candidates")
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_high_fidelity_ok_no_backtracking(self, mock_parse, mock_audit, mock_gen):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ] * 3)
        mock_audit.return_value = [
            _audit("p1", 0.96, {1: 0.97}),
        ]
        mock_gen.return_value = {}
        result = parse_instruction_llm("Turn left.", fallback_to_rules=False, vote_count=3)
        assert result["status"] == "ok"
        assert result["confidence"] == 0.96  # real score preserved, not forced to 1.0
        assert result["alternatives"] == []
        assert result.get("backtracking") == {"step_candidates": []}
        validate_result(result)

    @patch("vln_instruction_parser.llm.generate_step_candidates")
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_ok_with_backtracking(self, mock_parse, mock_audit, mock_gen):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "TURN", "direction": "right", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ] + [
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ])
        mock_audit.side_effect = lambda _inst, plans, **kw: [
            _audit("p1", 0.93, {1: 0.93}),
            _audit("p2", 0.85, {1: 0.85}),
        ] if len(plans) == 2 else [
            _audit(p["candidate_id"], 0.93, {1: 0.86})
            for p in plans
        ]
        mock_gen.return_value = {
            1: [{"action": "TURN", "direction": "right", "features": []}],
        }
        result = parse_instruction_llm("Turn left.", fallback_to_rules=False, vote_count=3)
        assert result["status"] == "ok"
        assert result["alternatives"] == []
        validate_result(result)

    @patch("vln_instruction_parser.llm.generate_step_candidates")
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    def test_step_margin_accepts_new_medium_boundary_when_plan_gate_passes(
        self, mock_audit, mock_gen
    ):
        plan = {
            "candidate_id": "p1",
            "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
            "constraints": [],
        }
        mock_audit.side_effect = [
            [_audit("p1", 0.93, {1: 0.95})],
            [_audit("rep_1_0", 0.84, {1: 0.55})],
        ]
        mock_gen.return_value = {
            1: [{"action": "TURN", "direction": "right", "features": []}],
        }

        result = _run_global_postprocess("Turn left.", [plan])

        candidates = result["backtracking"]["step_candidates"][0]["candidates"]
        assert result["status"] == "ok"
        assert candidates[0]["direction"] == "right"
        assert candidates[0]["confidence"] == 0.55

    @patch("vln_instruction_parser.llm.generate_step_candidates")
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    def test_relaxed_step_margin_does_not_relax_plan_gate(self, mock_audit, mock_gen):
        plan = {
            "candidate_id": "p1",
            "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
            "constraints": [],
        }
        mock_audit.side_effect = [
            [_audit("p1", 0.93, {1: 0.95})],
            [_audit("rep_1_0", 0.78, {1: 0.55})],
        ]
        mock_gen.return_value = {
            1: [{"action": "TURN", "direction": "right", "features": []}],
        }

        result = _run_global_postprocess("Turn left.", [plan])

        assert result["status"] == "ok"
        assert result["backtracking"] == {"step_candidates": []}

    @patch("vln_instruction_parser.llm.generate_step_candidates")
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    def test_relaxed_step_margin_still_rejects_blocked_candidate(self, mock_audit, mock_gen):
        plan = {
            "candidate_id": "p1",
            "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
            "constraints": [],
        }
        mock_audit.side_effect = [
            [_audit("p1", 0.93, {1: 0.95})],
            [_audit("rep_1_0", 0.84, {1: 0.55}, blocking=["wrong_direction"])],
        ]
        mock_gen.return_value = {
            1: [{"action": "TURN", "direction": "right", "features": []}],
        }

        result = _run_global_postprocess("Turn left.", [plan])

        assert result["status"] == "ok"
        assert result["backtracking"] == {"step_candidates": []}

    @patch("vln_instruction_parser.llm.generate_step_candidates")
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    def test_backtracking_ranks_only_best_two_candidate_steps(self, mock_audit, mock_gen):
        plan = {
            "candidate_id": "p1",
            "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
            "constraints": [],
        }
        mock_audit.side_effect = [
            [_audit("p1", 0.93, {1: 0.90})],
            [
                _audit("rep_1_0", 0.84, {1: 0.45}),
                _audit("rep_1_1", 0.84, {1: 0.65}),
                _audit("rep_1_2", 0.84, {1: 0.55}),
            ],
        ]
        mock_gen.return_value = {
            1: [
                {"action": "TURN", "direction": "right", "features": []},
                {"action": "TURN", "direction": "around", "features": []},
                {"action": "STOP", "features": []},
            ],
        }

        result = _run_global_postprocess("Turn left.", [plan])

        candidates = result["backtracking"]["step_candidates"][0]["candidates"]
        assert result["status"] == "ok"
        assert [candidate["rank"] for candidate in candidates] == [2, 3]
        assert [candidate["confidence"] for candidate in candidates] == [0.65, 0.55]

    @patch("vln_instruction_parser.llm.generate_step_candidates")
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_low_plan_confidence_needs_review(self, mock_parse, mock_audit, mock_gen):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "MOVE_FORWARD", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "STOP", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ])
        mock_audit.return_value = [
            _audit("p1", 0.87, {1: 0.88, 2: 0.85, 3: 0.87}),
        ]
        mock_gen.return_value = {}
        result = parse_instruction_llm("Go straight then turn left and stop.", fallback_to_rules=False, vote_count=3)
        assert result["status"] == "needs_review"
        assert result["alternatives"] == []
        validate_result(result)

    @patch("vln_instruction_parser.llm.generate_step_candidates")
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_blocking_issue_needs_review(self, mock_parse, mock_audit, mock_gen):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "MOVE_FORWARD", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ] * 2)
        mock_audit.return_value = [
            _audit("p1", 0.92, {1: 0.92}, blocking=["missing_condition"]),
        ]
        mock_gen.return_value = {}
        result = parse_instruction_llm("Go straight.", fallback_to_rules=False, vote_count=2)
        assert result["status"] == "needs_review"
        assert result["alternatives"] == []
        validate_result(result)

    @patch("vln_instruction_parser.llm.generate_step_candidates")
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_audit_unavailable_degradation(self, mock_parse, mock_audit, mock_gen):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ] * 2)
        mock_audit.return_value = None
        result = parse_instruction_llm("Turn left.", fallback_to_rules=False, vote_count=2)
        assert result["status"] == "needs_review"
        assert result["confidence"] == 0.0
        assert result.get("reason") == "instruction_fidelity_audit_unavailable"
        assert len(result["tasks"]) == 1
        assert mock_audit.call_count == 2
        assert result["alternatives"] == []
        validate_result(result)

    @patch("vln_instruction_parser.llm.generate_step_candidates")
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_audit_retry_success_on_short_path(self, mock_parse, mock_audit, mock_gen):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ])
        mock_audit.side_effect = [
            None,
            [_audit("p1", 0.97, {1: 0.97})],
        ]
        mock_gen.return_value = {}

        result = parse_instruction_llm("Turn left.", fallback_to_rules=False, vote_count=1)

        assert result["status"] == "ok"
        assert result["confidence"] == 0.97
        assert mock_audit.call_count == 2

    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_invalid_compiled_draft_reports_no_valid_candidate(self, mock_parse, mock_audit):
        mock_parse.return_value = (True, [{
            "actions": [{
                "id": "a1",
                "action": "TURN",
                "direction": "left",
                "features": [{"role": "where", "relation": "unknown_temporal", "landmark": "painting"}],
            }],
            "order": [],
            "constraints": [],
            "excluded": [],
        }])

        result = parse_instruction_llm(
            "Turn left before the painting.", fallback_to_rules=False, vote_count=1
        )

        assert result["status"] == "needs_review"
        assert result["tasks"] == []
        assert result["reason"] == "no_valid_candidate_plan"
        mock_audit.assert_not_called()

    @patch("vln_instruction_parser.parser.validate_result", side_effect=ValueError("invalid"))
    @patch("vln_instruction_parser.llm.generate_step_candidates", return_value={})
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    def test_final_validation_failure_is_identifiable(self, mock_audit, _mock_gen, _mock_validate):
        plan = {
            "candidate_id": "p1",
            "tasks": [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
            "constraints": [],
        }
        mock_audit.return_value = [_audit("p1", 0.97, {1: 0.97})]

        result = _run_global_postprocess("Turn left.", [plan])

        assert result["status"] == "needs_review"
        assert result["reason"] == "result_validation_failed"

    @patch("vln_instruction_parser.llm.generate_step_candidates")
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_backtracking_preserve_terminate_and_constraints(self, mock_parse, mock_audit, mock_gen):
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
        mock_audit.side_effect = lambda _inst, plans, **kw: [
            _audit("p1", 0.92, {1: 0.92}),
            _audit("p2", 0.83, {1: 0.83}),
        ] if len(plans) == 2 else [
            _audit(p["candidate_id"], 0.92, {1: 0.92})
            for p in plans
        ]
        mock_gen.return_value = {}
        result = parse_instruction_llm("Go to the door.", fallback_to_rules=False, vote_count=2)
        assert result["status"] == "needs_review"
        assert any(f.get("role") == "terminate" for f in result["tasks"][0].get("features", []))
        assert len(result["constraints"]) == 1
        assert result["alternatives"] == []
        validate_result(result)

    @patch("vln_instruction_parser.llm.generate_step_candidates")
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_three_different_votes_all_audited(self, mock_parse, mock_audit, mock_gen):
        """All distinct plans enter fidelity audit."""
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "MOVE_FORWARD", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
            {"actions": [{"id": "a1", "action": "STOP", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ])
        mock_audit.return_value = [
            _audit("p1", 0.88, {1: 0.88}),
            _audit("p2", 0.75, {1: 0.75}),
            _audit("p3", 0.70, {1: 0.70}),
        ]
        mock_gen.return_value = {}
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
    @patch("vln_instruction_parser.llm.generate_step_candidates")
    @patch("vln_instruction_parser.llm.audit_plans_against_instruction")
    @patch("vln_instruction_parser.llm.parse_with_llm")
    def test_output_has_no_raw_text_or_candidate_id(self, mock_parse, mock_audit, mock_gen):
        mock_parse.return_value = (True, [
            {"actions": [{"id": "a1", "action": "TURN", "direction": "left", "features": []}],
             "order": [], "constraints": [], "excluded": []},
        ] * 3)
        mock_audit.return_value = [
            {"candidate_id": "p1", "plan_confidence": 0.96, "blocking_issues": [],
             "step_confidences": [{"step_id": 1, "confidence": 0.97}]},
        ]
        mock_gen.return_value = {}
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
        # Test the rule interface directly; auto intentionally invokes the LLM.
        from vln_instruction_parser.parser import parse_instruction
        result = parse_instruction("Turn left.")
        assert result["status"] == "ok"
        assert result["confidence"] == 0.85
        assert result["alternatives"] == []
