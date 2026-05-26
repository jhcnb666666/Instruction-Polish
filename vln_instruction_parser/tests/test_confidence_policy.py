"""Tests for plan-level confidence policy."""

import pytest
from vln_instruction_parser.parser import apply_plan_confidence_policy


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
        assert result["backtracking"] == {}


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
