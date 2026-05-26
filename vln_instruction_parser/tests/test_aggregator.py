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
        """3 identical votes -> single candidate with vote_support=3."""
        actions = [
            {"id": "a1", "action": "MOVE_FORWARD", "direction": "straight", "features": [{"role": "path", "relation": "along", "landmark": "hallway"}], "confidence": 0.92},
            {"id": "a2", "action": "TURN", "direction": "left", "features": [{"role": "where", "relation": "at", "landmark": "sofa"}], "confidence": 0.95},
        ]
        vote = _wrap(actions)
        ranked = aggregate_votes([vote, vote, vote], "test instruction")
        assert len(ranked) == 1
        assert ranked[0]["candidate_id"] == "p1"
        assert ranked[0]["vote_support"] == 3
        assert len(ranked[0]["tasks"]) == 2
        t1 = ranked[0]["tasks"][0]
        assert t1["action"] == "MOVE_FORWARD"

    def test_majority_wins_different_action_counts(self):
        """2 votes with 2 actions, 1 vote with 3 actions -> majority 2 actions is top candidate."""
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
        ranked = aggregate_votes(
            [_wrap(vote_a), _wrap(vote_b), _wrap(vote_c)], "test"
        )
        assert ranked[0]["vote_support"] == 2
        assert len(ranked[0]["tasks"]) == 2

    def test_no_majority_keeps_all(self):
        """All votes different -> keep up to 3 candidates."""
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
        ranked = aggregate_votes(
            [_wrap(vote_a), _wrap(vote_b), _wrap(vote_c)], "test"
        )
        assert len(ranked) == 3
        assert ranked[0]["candidate_id"] == "p1"
        assert ranked[1]["candidate_id"] == "p2"
        assert ranked[2]["candidate_id"] == "p3"

    def test_two_of_three_agree_action(self):
        """2 votes say TURN, 1 says MOVE_FORWARD -> majority TURN is top candidate."""
        vote_a = [{"id": "a1", "action": "TURN", "direction": "left", "features": [{"role": "where", "relation": "at", "landmark": "sofa"}], "confidence": 0.92}]
        vote_b = [{"id": "a1", "action": "TURN", "direction": "left", "features": [{"role": "where", "relation": "at", "landmark": "sofa"}], "confidence": 0.90}]
        vote_c = [{"id": "a1", "action": "MOVE_FORWARD", "direction": "left", "features": [{"role": "where", "relation": "at", "landmark": "sofa"}], "confidence": 0.85}]
        ranked = aggregate_votes(
            [_wrap(vote_a), _wrap(vote_b), _wrap(vote_c)], "test"
        )
        assert ranked[0]["vote_support"] == 2
        t1 = ranked[0]["tasks"][0]
        assert t1["action"] == "TURN"

    def test_three_different_execution_orders(self):
        """Three different plans -> all three kept as candidates."""
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
        ranked = aggregate_votes(
            [_wrap(vote_a), _wrap(vote_b), _wrap(vote_c)], "test"
        )
        assert len(ranked) == 3
        assert ranked[0]["vote_support"] == 1

    def test_majority_sequence_used(self):
        """2/3 agree on sequence -> top candidate is majority plan."""
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
        ranked = aggregate_votes(
            [_wrap(vote_a), _wrap(vote_b), _wrap(vote_c)], "test"
        )
        assert ranked[0]["vote_support"] == 2
        assert len(ranked[0]["tasks"]) == 2
        assert ranked[0]["tasks"][0]["action"] == "MOVE_FORWARD"
        assert ranked[0]["tasks"][1]["action"] == "TURN"

    def test_empty_votes(self):
        ranked = aggregate_votes([], "test")
        assert ranked == []
