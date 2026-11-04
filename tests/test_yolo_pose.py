"""Inference-size handling in yolo_pose (no model needed)."""

import numpy as np
import pytest

from yolo_pose import parse_image_size, resolve_image_size


@pytest.mark.parametrize("value", [None, "native", "NATIVE", " native ", "", "0", 0, "none"])
def test_parse_native(value):
    assert parse_image_size(value) is None


@pytest.mark.parametrize("value, expected", [("640", 640), (640, 640), ("1280", 1280)])
def test_parse_integer(value, expected):
    assert parse_image_size(value) == expected


@pytest.mark.parametrize("value", ["big", "-1", -1, "6.4", 1.5, True])
def test_parse_rejects_bad_values(value):
    with pytest.raises(ValueError):
        parse_image_size(value)


def test_native_size_is_frame_size_rounded_up_to_stride():
    frame = np.zeros((704, 1280, 3), np.uint8)
    assert resolve_image_size(None, frame) == [704, 1280]
    frame = np.zeros((720, 1280, 3), np.uint8)
    assert resolve_image_size(None, frame) == [736, 1280]
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert resolve_image_size(None, frame) == [1088, 1920]


def test_integer_size_passes_through():
    frame = np.zeros((1080, 1920, 3), np.uint8)
    assert resolve_image_size(640, frame) == 640


def test_settings_read_image_size(monkeypatch):
    from backend.config import Settings

    monkeypatch.setenv("FALL_IMAGE_SIZE", "640")
    assert Settings().image_size == 640
    monkeypatch.setenv("FALL_IMAGE_SIZE", "native")
    assert Settings().image_size is None
    monkeypatch.delenv("FALL_IMAGE_SIZE")
    assert Settings().image_size is None
