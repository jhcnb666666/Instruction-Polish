"""Tests for normalizer module."""

import pytest
from vln_instruction_parser.normalizer import (
    normalize_action,
    normalize_direction,
    normalize_relation,
    extract_landmark,
    has_3d_phrase,
    is_common_vln_pattern,
)


class TestNormalizeAction:
    def test_walk(self):
        action, conf = normalize_action("walk to the door")
        assert action == "MOVE_FORWARD"
        assert conf > 0

    def test_turn(self):
        action, conf = normalize_action("turn left")
        assert action == "TURN"

    def test_stop(self):
        action, conf = normalize_action("stop here")
        assert action == "STOP"

    def test_enter(self):
        action, conf = normalize_action("enter the room")
        assert action == "ENTER"

    def test_unknown(self):
        action, conf = normalize_action("foo bar")
        assert action == "UNKNOWN"

    def test_3d_phrase_action(self):
        action, conf = normalize_action("go upstairs")
        assert action == "UNKNOWN"


class TestNormalizeDirection:
    def test_left(self):
        d, conf = normalize_direction("turn left")
        assert d == "left"

    def test_right(self):
        d, conf = normalize_direction("turn right")
        assert d == "right"

    def test_forward(self):
        d, conf = normalize_direction("go forward")
        assert d == "forward"

    def test_back(self):
        d, conf = normalize_direction("go back")
        assert d == "backward"

    def test_around(self):
        d, conf = normalize_direction("turn around")
        assert d == "around"

    def test_unknown(self):
        d, conf = normalize_direction("enter the room")
        assert d == "unknown"


class TestNormalizeRelation:
    def test_in_front_of(self):
        r, conf = normalize_relation("in front of the sofa")
        assert r == "in_front_of"

    def test_near(self):
        r, conf = normalize_relation("by the kitchen door")
        assert r == "near"

    def test_toward(self):
        r, conf = normalize_relation("walk toward the door")
        assert r == "toward"

    def test_between(self):
        r, conf = normalize_relation("move between two chairs")
        assert r == "between"

    def test_unknown(self):
        r, conf = normalize_relation("turn left")
        assert r == "unknown"


class TestExtractLandmark:
    def test_basic(self):
        lm, conf = extract_landmark("walk to the kitchen", "toward", "GO_TO")
        assert lm == "kitchen"

    def test_hallway(self):
        lm, conf = extract_landmark("go down the hallway", "along", "MOVE_FORWARD")
        assert "hallway" in lm


class TestHas3DPhrase:
    def test_upstairs(self):
        assert has_3d_phrase("go upstairs") is True

    def test_downstairs(self):
        assert has_3d_phrase("run downstairs") is True

    def test_elevator(self):
        assert has_3d_phrase("take the elevator") is True

    def test_floor(self):
        assert has_3d_phrase("move to the second floor") is True

    def test_climb(self):
        assert has_3d_phrase("climb the stairs") is True

    def test_elevator_as_landmark(self):
        # "past the elevator" is 2D, not 3D
        assert has_3d_phrase("past the elevator") is False
        assert has_3d_phrase("near the elevator") is False
        assert has_3d_phrase("wait by the elevator") is False

    def test_take_elevator_is_3d(self):
        assert has_3d_phrase("take the elevator") is True
        assert has_3d_phrase("ride the lift") is True
        assert has_3d_phrase("use the elevator") is True

    def test_2d_ok(self):
        assert has_3d_phrase("walk to the kitchen") is False


class TestCommonVLNPattern:
    def test_walk_to(self):
        assert is_common_vln_pattern("walk to the table") is True

    def test_turn_left(self):
        assert is_common_vln_pattern("turn left") is True

    def test_go_straight(self):
        assert is_common_vln_pattern("go straight") is True

    def test_not_common(self):
        assert is_common_vln_pattern("foobar") is False
