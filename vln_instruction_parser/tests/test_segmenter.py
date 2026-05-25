"""Tests for instruction segmentation."""

import pytest
from vln_instruction_parser.segmenter import segment_instruction, normalize_whitespace, split_by_comma_clauses


class TestNormalizeWhitespace:
    def test_basic(self):
        assert normalize_whitespace("  hello   world  ") == "hello world"

    def test_newlines(self):
        assert normalize_whitespace("hello\n\n\tworld") == "hello world"


class TestSplitByCommaClauses:
    def test_no_split(self):
        assert split_by_comma_clauses("turn left in front of the sofa") == ["turn left in front of the sofa"]

    def test_split_action_clauses(self):
        result = split_by_comma_clauses("go straight, turn left")
        assert result == ["go straight", "turn left"]

    def test_no_split_non_action(self):
        result = split_by_comma_clauses("the kitchen, which is near the door")
        assert result == ["the kitchen, which is near the door"]


class TestSegmentInstruction:
    def test_then_split(self):
        text = "Go down the hallway, then turn left"
        segs = segment_instruction(text)
        assert len(segs) == 2
        assert segs[0] == "Go down the hallway"
        assert segs[1] == "turn left"

    def test_and_then_split(self):
        text = "Walk to the table and then turn right"
        segs = segment_instruction(text)
        assert len(segs) == 2
        assert segs[0] == "Walk to the table"
        assert segs[1] == "turn right"

    def test_comma_split(self):
        text = "Go down the hallway, turn left in front of the sofa, then stop by the kitchen door"
        segs = segment_instruction(text)
        assert len(segs) == 3
        assert segs[0] == "Go down the hallway"
        assert "turn left" in segs[1]
        assert "stop" in segs[2]

    def test_sentence_boundary(self):
        text = "Walk to the table. Turn right. Stop."
        segs = segment_instruction(text)
        assert len(segs) == 3
        assert segs[0] == "Walk to the table"
        assert segs[1] == "Turn right"
        assert segs[2] == "Stop"

    def test_next_finally(self):
        text = "Enter the room. Next, face the door. Finally, stop."
        segs = segment_instruction(text)
        assert len(segs) >= 3
        assert any("Enter the room" in s for s in segs)
        assert any("face the door" in s for s in segs)
        assert any("stop" in s for s in segs)

    def test_empty(self):
        assert segment_instruction("") == []

    def test_single_segment(self):
        assert segment_instruction("Go to the kitchen") == ["Go to the kitchen"]
