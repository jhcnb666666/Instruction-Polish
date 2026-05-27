"""Tests for plan-level confidence policy."""

import pytest
from vln_instruction_parser.parser import (
    apply_plan_confidence_policy,
    apply_step_candidate_policy,
)


def _plan(confidence, tasks=None, constraints=None):
    return {
        "status": "ok",
        "confidence": confidence,
        "tasks": tasks if tasks is not None else [{"step_id": 1, "action": "TURN", "direction": "left", "features": []}],
        "constraints": constraints if constraints is not None else [],
    }


class TestConfidencePolicyTier1:
    def test_c1_above_095(self):
        plans = [_plan(0.96)]
        result = apply_plan_confidence_policy(plans)
        assert result["confidence"] == 1.0
        assert result["status"] == "ok"
        assert result["alternatives"] == []
        assert result["backtracking"] == {"step_candidates": []}


class TestConfidencePolicyTier2:
    def test_c1_093_c2_084_no_c3(self):
        plans = [_plan(0.93), _plan(0.84)]
        result = apply_plan_confidence_policy(plans)
        # Close competitor at plan level => needs_review until localized
        assert result["status"] == "needs_review"
        assert result["alternatives"] == []

    def test_c1_093_c2_084_c3_082(self):
        plans = [_plan(0.93), _plan(0.84), _plan(0.82)]
        result = apply_plan_confidence_policy(plans)
        assert result["status"] == "needs_review"
        assert result["alternatives"] == []

    def test_c1_093_c2_far_075(self):
        plans = [_plan(0.93), _plan(0.75)]
        result = apply_plan_confidence_policy(plans)
        assert result["status"] == "ok"
        assert result["alternatives"] == []

    def test_c1_090_boundary(self):
        # C1 == 0.90 falls into tier 3
        plans = [_plan(0.90), _plan(0.72)]
        result = apply_plan_confidence_policy(plans)
        assert result["status"] == "needs_review"
        assert result["alternatives"] == []


class TestConfidencePolicyTier3:
    def test_c1_087_c2_070_c3_063(self):
        plans = [_plan(0.87), _plan(0.70), _plan(0.63)]
        result = apply_plan_confidence_policy(plans)
        assert result["status"] == "needs_review"
        assert result["alternatives"] == []

    def test_c1_087_c2_066_c3_060(self):
        plans = [_plan(0.87), _plan(0.66), _plan(0.60)]
        result = apply_plan_confidence_policy(plans)
        assert result["status"] == "needs_review"
        assert result["alternatives"] == []

    def test_empty_plans(self):
        result = apply_plan_confidence_policy([])
        assert result["status"] == "none"
        assert result["confidence"] == 0.0
        assert result["reason"] == "no_action"


def _candidate(direction):
    return {"action": "TURN", "direction": direction, "features": []}


class TestStepCandidatePolicy:
    def test_above_095_ignores_candidates(self):
        result = apply_step_candidate_policy(
            [{"step_id": 1, "confidence": 0.96}],
            [{"step_id": 1, "rank": 2, "confidence": 0.96}],
            {1: [_candidate("right")]},
        )
        assert result == {"step_candidates": []}

    def test_medium_tier_includes_040_boundary_and_rejects_beyond_it(self):
        result = apply_step_candidate_policy(
            [{"step_id": 1, "confidence": 0.95}],
            [
                {"step_id": 1, "rank": 2, "confidence": 0.55},
                {"step_id": 1, "rank": 3, "confidence": 0.54},
            ],
            {1: [_candidate("right"), _candidate("around")]},
        )
        candidates = result["step_candidates"][0]["candidates"]
        assert [candidate["confidence"] for candidate in candidates] == [0.55]

    def test_low_tier_includes_050_boundary_and_rejects_beyond_it(self):
        result = apply_step_candidate_policy(
            [{"step_id": 1, "confidence": 0.90}],
            [
                {"step_id": 1, "rank": 2, "confidence": 0.40},
                {"step_id": 1, "rank": 3, "confidence": 0.39},
            ],
            {1: [_candidate("right"), _candidate("around")]},
        )
        candidates = result["step_candidates"][0]["candidates"]
        assert [candidate["confidence"] for candidate in candidates] == [0.40]

    def test_saves_only_rank_two_and_three(self):
        result = apply_step_candidate_policy(
            [{"step_id": 1, "confidence": 0.90}],
            [
                {"step_id": 1, "rank": 2, "confidence": 0.70},
                {"step_id": 1, "rank": 3, "confidence": 0.65},
                {"step_id": 1, "rank": 4, "confidence": 0.64},
            ],
            {1: [_candidate("right"), _candidate("around"), _candidate("left")]},
        )
        candidates = result["step_candidates"][0]["candidates"]
        assert [candidate["rank"] for candidate in candidates] == [2, 3]
