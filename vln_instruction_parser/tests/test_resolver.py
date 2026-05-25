"""Tests for context resolution."""

from vln_instruction_parser.resolver import resolve_context_for_candidates, update_context


class TestResolverNoInheritance:
    def test_no_landmark_inheritance(self):
        """Resolver must NOT auto-fill missing landmarks from context."""
        candidates = [{"action": "TURN", "direction": "right", "features": [], "confidence": 0.9}]
        context = {"landmark": "table"}
        resolved = resolve_context_for_candidates(candidates, context)
        assert resolved[0]["features"] == []

    def test_update_context_returns_empty(self):
        task = {"step_id": 1, "action": "MOVE_FORWARD", "features": [{"role": "where", "relation": "at", "landmark": "table"}]}
        ctx = update_context({}, task)
        assert ctx == {}
