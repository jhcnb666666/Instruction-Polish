"""Tests for vote aggregation logic — compact plan aggregation."""

import pytest
from vln_instruction_parser.aggregator import aggregate_votes, normalize_landmark


def _task(step_id, action, direction=None, features=None, confidence=0.9):
    t = {
        "step_id": step_id,
        "action": action,
        "features": features if features is not None else [],
    }
    if direction is not None:
        t["direction"] = direction
    return t


def _wrap(actions):
    """Wrap action list into a raw vote dict."""
    return {
        "actions": actions,
        "order": [],
        "constraints": [],
        "excluded": [],
    }


class TestNormalizeLandmark:
    def test_basic(self):
        assert normalize_landmark("  Sofa  ") == "sofa"
        assert normalize_landmark("The KITCHEN") == "the kitchen"


class TestAggregateVotes:
    def test_all_votes_agree(self):
        """3 identical votes -> high confidence, no alternatives."""
        actions = [
            {"id": "a1", "action": "MOVE_FORWARD", "direction": "straight", "features": [{"role": "path", "relation": "along", "landmark": "hallway"}], "confidence": 0.92},
            {"id": "a2", "action": "TURN", "direction": "left", "features": [{"role": "where", "relation": "at", "landmark": "sofa"}], "confidence": 0.95},
        ]
        vote = _wrap(actions)
        success, ranked_plans, needs_adj = aggregate_votes(
            [vote, vote, vote], "test instruction"
        )
        assert success is True
        assert needs_adj is False
        assert len(ranked_plans) == 1
        assert len(ranked_plans[0]["tasks"]) == 2
        t1 = ranked_plans[0]["tasks"][0]
        assert t1["action"] == "MOVE_FORWARD"
        assert ranked_plans[0]["confidence"] >= 0.90

    def test_majority_wins_different_action_counts(self):
        """2 votes with 2 actions, 1 vote with 3 actions -> majority 2 actions used."""
        vote_a = [
            {"id": "a1", "action": "MOVE_FORWARD", "features": [], "confidence": 0.9},
            {"id": "a2", "action": "TURN", "direction": "left", "features": [], "confidence": 0.9},
        ]
        vote_b = [
            {"id": "a1", "action": "MOVE_FORWARD", "features": [], "confidence": 0.9},
            {"id": "a2", "action": "TURN", "direction": "left", "features": [], "confidence": 0.9},
        ]
        vote_c = [
            {"id": "a1", "action": "MOVE_FORWARD", "features": [], "confidence": 0.9},
            {"id": "a2", "action": "TURN", "direction": "left", "features": [], "confidence": 0.9},
            {"id": "a3", "action": "STOP", "features": [], "confidence": 0.9},
        ]
        success, ranked_plans, needs_adj = aggregate_votes(
            [_wrap(vote_a), _wrap(vote_b), _wrap(vote_c)], "test"
        )
        assert success is True
        assert needs_adj is False
        assert len(ranked_plans[0]["tasks"]) == 2

    def test_no_majority_action_count(self):
        """All votes have different action counts -> fallback."""
        vote_a = [{"id": "a1", "action": "MOVE_FORWARD", "features": [], "confidence": 0.9}]
        vote_b = [
            {"id": "a1", "action": "MOVE_FORWARD", "features": [], "confidence": 0.9},
            {"id": "a2", "action": "TURN", "features": [], "confidence": 0.9},
        ]
        vote_c = [
            {"id": "a1", "action": "MOVE_FORWARD", "features": [], "confidence": 0.9},
            {"id": "a2", "action": "TURN", "features": [], "confidence": 0.9},
            {"id": "a3", "action": "STOP", "features": [], "confidence": 0.9},
        ]
        success, ranked_plans, needs_adj = aggregate_votes(
            [_wrap(vote_a), _wrap(vote_b), _wrap(vote_c)], "test"
        )
        assert success is False

    def test_two_of_three_agree_action(self):
        """2 votes say TURN, 1 says MOVE_FORWARD -> majority TURN wins."""
        vote_a = [{"id": "a1", "action": "TURN", "direction": "left", "features": [{"role": "where", "relation": "at", "landmark": "sofa"}], "confidence": 0.92}]
        vote_b = [{"id": "a1", "action": "TURN", "direction": "left", "features": [{"role": "where", "relation": "at", "landmark": "sofa"}], "confidence": 0.90}]
        vote_c = [{"id": "a1", "action": "MOVE_FORWARD", "direction": "left", "features": [{"role": "where", "relation": "at", "landmark": "sofa"}], "confidence": 0.85}]
        success, ranked_plans, needs_adj = aggregate_votes(
            [_wrap(vote_a), _wrap(vote_b), _wrap(vote_c)], "test"
        )
        assert success is True
        t1 = ranked_plans[0]["tasks"][0]
        assert t1["action"] == "TURN"

    def test_sequence_conflict_needs_adjudication(self):
        """Three different execution orders -> signal adjudication needed."""
        vote_a = [
            {"id": "a1", "action": "MOVE_FORWARD", "features": [], "confidence": 0.9},
            {"id": "a2", "action": "TURN", "features": [], "confidence": 0.9},
        ]
        vote_b = [
            {"id": "a1", "action": "TURN", "features": [], "confidence": 0.9},
            {"id": "a2", "action": "MOVE_FORWARD", "features": [], "confidence": 0.9},
        ]
        vote_c = [
            {"id": "a1", "action": "GO_TO", "features": [], "confidence": 0.9},
            {"id": "a2", "action": "STOP", "features": [], "confidence": 0.9},
        ]
        success, ranked_plans, needs_adj = aggregate_votes(
            [_wrap(vote_a), _wrap(vote_b), _wrap(vote_c)], "test"
        )
        assert success is False
        assert needs_adj is True

    def test_majority_sequence_used_no_adjudication(self):
        """2/3 agree on sequence -> use majority, no adjudication needed."""
        vote_a = [
            {"id": "a1", "action": "MOVE_FORWARD", "features": [], "confidence": 0.9},
            {"id": "a2", "action": "TURN", "features": [], "confidence": 0.9},
        ]
        vote_b = [
            {"id": "a1", "action": "MOVE_FORWARD", "features": [], "confidence": 0.9},
            {"id": "a2", "action": "TURN", "features": [], "confidence": 0.9},
        ]
        vote_c = [
            {"id": "a1", "action": "TURN", "features": [], "confidence": 0.9},
            {"id": "a2", "action": "MOVE_FORWARD", "features": [], "confidence": 0.9},
        ]
        success, ranked_plans, needs_adj = aggregate_votes(
            [_wrap(vote_a), _wrap(vote_b), _wrap(vote_c)], "test"
        )
        assert success is True
        assert needs_adj is False
        assert len(ranked_plans[0]["tasks"]) == 2
        assert ranked_plans[0]["tasks"][0]["action"] == "MOVE_FORWARD"
        assert ranked_plans[0]["tasks"][1]["action"] == "TURN"

    def test_empty_votes(self):
        success, ranked_plans, needs_adj = aggregate_votes([], "test")
        assert success is False
