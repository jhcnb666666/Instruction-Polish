"""Tests for semantic compiler."""

import pytest
from vln_instruction_parser.semantic_compiler import compile_draft


class TestCompileDraftBasic:
    def test_empty_draft(self):
        result = compile_draft({})
        assert result["status"] == "ok"
        assert result["tasks"] == []

    def test_single_action(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "TURN", "direction": "left", "features": [], "confidence": 0.95}
            ],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        assert result["status"] == "ok"
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["action"] == "TURN"
        assert result["tasks"][0]["direction"] == "left"

    def test_topological_sort(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "TURN", "direction": "left", "features": [], "confidence": 0.9},
                {"id": "a2", "action": "MOVE_FORWARD", "features": [], "confidence": 0.9},
            ],
            "order": [{"before": "a2", "after": "a1"}],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        assert result["tasks"][0]["action"] == "MOVE_FORWARD"
        assert result["tasks"][1]["action"] == "TURN"

    def test_prune_excluded(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "ENTER", "features": [{"role": "where", "relation": "into", "landmark": "kitchen"}], "confidence": 0.9},
                {"id": "a2", "action": "TURN", "direction": "right", "features": [{"role": "where", "relation": "at", "landmark": "door"}], "confidence": 0.9},
            ],
            "order": [],
            "constraints": [],
            "excluded": ["a1"],
        }
        result = compile_draft(draft)
        assert len(result["tasks"]) == 1
        assert result["tasks"][0]["action"] == "TURN"

    def test_constraint_conversion(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "WAIT", "features": [{"role": "where", "relation": "outside", "landmark": "door"}], "confidence": 0.88}
            ],
            "order": [],
            "constraints": [
                {"type": "forbidden_action", "action": "ENTER", "features": [{"role": "where", "relation": "inside", "landmark": "room"}]}
            ],
            "excluded": [],
        }
        result = compile_draft(draft)
        assert len(result["constraints"]) == 1
        assert result["constraints"][0]["type"] == "forbidden_action"
        assert result["constraints"][0]["action"] == "ENTER"


class TestCompileDraftFeatures:
    def test_terminate_feature(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "MOVE_FORWARD", "direction": "straight", "features": [
                    {"role": "terminate", "trigger": "see", "relation": "left_of_agent", "landmark": "sofa"}
                ], "confidence": 0.9}
            ],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        f = result["tasks"][0]["features"][0]
        assert f["role"] == "terminate"
        assert f["trigger"] == "see"
        assert f["landmark"] == "sofa"

    def test_start_feature(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "TURN", "direction": "left", "features": [
                    {"role": "start", "trigger": "see", "landmark": "sofa"}
                ], "confidence": 0.9}
            ],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        f = result["tasks"][0]["features"][0]
        assert f["role"] == "start"
        assert f["trigger"] == "see"

    def test_invalid_feature_role(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "TURN", "features": [{"role": "invalid"}], "confidence": 0.9}
            ],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        assert result["status"] == "ok"
        assert result["tasks"][0]["features"] == []

    def test_terminate_missing_trigger(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "MOVE_FORWARD", "features": [{"role": "terminate", "landmark": "sofa"}], "confidence": 0.9}
            ],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        assert result["status"] == "ok"
        assert result["tasks"][0]["features"] == []

    def test_terminate_unknown_relation_keeps_executable_trigger(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "MOVE_FORWARD", "features": [
                    {
                        "role": "terminate",
                        "trigger": "notice",
                        "relation": "change_in",
                        "landmark": "floor tiles color from white to grey",
                    }
                ], "confidence": 0.9}
            ],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        assert result["status"] == "ok"
        assert result["tasks"][0]["features"] == [{
            "role": "terminate",
            "landmark": "floor tiles color from white to grey",
            "trigger": "notice",
        }]

    def test_before_reaching_alias_is_preserved(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "TURN", "direction": "left", "features": [
                    {"role": "where", "relation": "before_reaching", "landmark": "painting"}
                ], "confidence": 0.9}
            ],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        assert result["status"] == "ok"
        assert result["tasks"][0]["features"] == [
            {"role": "where", "relation": "before", "landmark": "painting"}
        ]

    def test_immediately_before_alias_is_just_before(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "TURN", "direction": "left", "features": [
                    {"role": "where", "relation": "immediately_before", "landmark": "grey tiles"}
                ], "confidence": 0.9}
            ],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        assert result["status"] == "ok"
        assert result["tasks"][0]["features"] == [
            {"role": "where", "relation": "just_before", "landmark": "grey tiles"}
        ]

    def test_unknown_relation_invalidates_candidate(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "TURN", "direction": "right", "features": [
                    {"role": "where", "relation": "around_the_corner_from", "landmark": "glass wall"}
                ], "confidence": 0.9}
            ],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        assert result["status"] == "needs_review"
        assert result["tasks"] == []
        assert result["reason"] == "invalid_feature_relation:around_the_corner_from"


class TestCompileDraftVertical:
    def test_go_upstairs_unsupported(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "MOVE_FORWARD", "features": [{"role": "target", "relation": "toward", "landmark": "upstairs"}], "confidence": 0.9}
            ],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        # "upstairs" in landmark with go action should be flagged
        assert result["status"] == "unsupported"

    def test_walk_past_stairs_ok(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "PASS", "features": [{"role": "progress", "relation": "past", "landmark": "stairs"}], "confidence": 0.9}
            ],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        assert result["status"] == "ok"

    def test_stop_beside_elevator_ok(self):
        draft = {
            "actions": [
                {"id": "a1", "action": "STOP", "features": [{"role": "where", "relation": "near", "landmark": "elevator"}], "confidence": 0.9}
            ],
            "order": [],
            "constraints": [],
            "excluded": [],
        }
        result = compile_draft(draft)
        assert result["status"] == "ok"
