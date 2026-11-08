import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sliding_window import TrackWindows, normalize_window  # noqa: E402

SAMPLE_JSON = REPO_ROOT / "data" / "output" / "001.json"

# Windows are float32 (about 7 significant digits), so compare with a small
# absolute tolerance instead of NumPy's default atol=0.
TOL = {"rtol": 1e-6, "atol": 1e-6}


def make_person(track_id, x=100.0, y=200.0, height=100.0, width=40.0, conf=0.9):
    """A fake person whose 12 body keypoints all sit at the box center."""
    return {
        "track_id": track_id,
        "bbox": [x - width / 2, y - height / 2, x + width / 2, y + height / 2],
        "confidence": 0.9,
        "keypoints": [{"x": x, "y": y, "confidence": conf}] * 12,
    }


def confidences(window):
    """Joint 0's confidence per frame. Tests use it to tag which frame is which."""
    return window.keypoints[:, 0, 2]


# Sliding


def test_window_keeps_newest_frames_in_order():
    windows = TrackWindows(window_size=3)
    for frame in range(5):
        windows.update(frame, [make_person(1, x=100.0 + frame, conf=frame / 10)])

    window = windows.window(1)
    assert window.is_full
    assert window.end_frame == 4
    assert window.keypoints.shape == (3, 12, 3)
    # Frames 2, 3, 4 remain, oldest first.
    np.testing.assert_allclose(confidences(window), [0.2, 0.3, 0.4], **TOL)
    # x is relative to the newest box center, scaled by box height (100).
    np.testing.assert_allclose(window.keypoints[:, 0, 0], [-0.02, -0.01, 0.0], **TOL)


def test_flatten_puts_each_joints_x_y_conf_side_by_side():
    windows = TrackWindows(window_size=2)
    windows.update(0, [make_person(1)])
    window = windows.window(1)

    flat = window.flatten()
    assert flat.shape == (2, 36)
    assert flat.dtype == np.float32
    np.testing.assert_array_equal(flat[1, 3:6], window.keypoints[1, 1])


def test_short_track_is_left_padded_and_masked():
    windows = TrackWindows(window_size=4)
    windows.update(0, [make_person(1, conf=0.5)])
    windows.update(1, [make_person(1, conf=0.6)])

    window = windows.window(1)
    assert not window.is_full
    assert window.mask.tolist() == [False, False, True, True]
    assert not window.keypoints[:2].any()
    assert not window.box_features[:2].any()
    np.testing.assert_allclose(confidences(window)[2:], [0.5, 0.6], **TOL)


def test_returned_window_is_a_copy():
    windows = TrackWindows(window_size=2)
    windows.update(0, [make_person(1)])
    windows.window(1).keypoints[:] = 99.0

    assert windows.window(1).keypoints.max() < 99.0


# Separating people


def test_people_do_not_share_windows():
    windows = TrackWindows(window_size=2)
    windows.update(0, [make_person(1, conf=0.1), make_person(2, conf=0.2)])
    windows.update(1, [make_person(1, conf=0.3), make_person(2, conf=0.4)])

    np.testing.assert_allclose(confidences(windows.window(1)), [0.1, 0.3], **TOL)
    np.testing.assert_allclose(confidences(windows.window(2)), [0.2, 0.4], **TOL)


def test_untracked_people_are_skipped():
    windows = TrackWindows(window_size=2)
    updated = windows.update(0, [make_person(1), make_person(None)])

    assert updated == [1]
    assert len(windows) == 1
    assert windows.window(2) is None


# Gaps and stale tracks


@pytest.mark.parametrize(
    ("missing", "expected_frames"),
    [
        (0, 3),  # no gap
        (1, 4),
        (2, 5),  # exactly max_gap: filled
        (3, 1),  # one past max_gap: window restarts
    ],
)
def test_gap_is_filled_up_to_max_gap(missing, expected_frames):
    windows = TrackWindows(window_size=10, max_gap=2)
    windows.update(0, [make_person(1)])
    windows.update(1, [make_person(1)])
    for frame in range(2, 2 + missing):
        windows.update(frame, [])
    windows.update(2 + missing, [make_person(1)])

    assert windows.window(1).mask.sum() == expected_frames


def test_gap_carries_last_pose_forward_with_zero_confidence():
    windows = TrackWindows(window_size=4, max_gap=2)
    windows.update(0, [make_person(1, x=100.0, conf=0.9)])
    windows.update(3, [make_person(1, x=100.0, conf=0.8)])

    window = windows.window(1)
    assert window.is_full
    np.testing.assert_allclose(confidences(window), [0.9, 0.0, 0.0, 0.8], **TOL)
    # The filled frames repeat the last pose and box exactly.
    np.testing.assert_array_equal(window.keypoints[1, :, :2], window.keypoints[0, :, :2])
    np.testing.assert_array_equal(window.box_features[1], window.box_features[0])


def test_stale_track_is_dropped_on_an_empty_frame():
    windows = TrackWindows(window_size=3, max_gap=2, stale_after=5)
    windows.update(0, [make_person(1)])
    windows.update(5, [])
    assert 1 in windows
    windows.update(6, [])
    assert 1 not in windows
    assert windows.window(1) is None


def test_returning_stale_track_starts_a_fresh_window():
    windows = TrackWindows(window_size=3, max_gap=2, stale_after=5)
    windows.update(0, [make_person(1)])
    windows.update(1, [make_person(1)])
    windows.update(20, [make_person(1)])

    window = windows.window(1)
    assert window.mask.sum() == 1
    assert window.end_frame == 20


# Bad input


@pytest.mark.parametrize(
    "people",
    [
        [make_person(1), make_person(1)],  # duplicate ID in one frame
        [make_person(2), {**make_person(3), "keypoints": [{"x": 0, "y": 0, "confidence": 0}] * 5}],
        [make_person(2), {**make_person(3), "bbox": [0, 0, 10]}],
    ],
    ids=["duplicate-id", "wrong-keypoint-count", "bad-bbox"],
)
def test_bad_frame_is_rejected_without_partial_changes(people):
    windows = TrackWindows(window_size=3)
    windows.update(0, [make_person(1)])

    with pytest.raises(ValueError):
        windows.update(1, people)
    assert len(windows) == 1
    assert windows.window(1).end_frame == 0


def test_frames_out_of_order_are_rejected():
    windows = TrackWindows(window_size=3)
    windows.update(5, [make_person(1)])
    with pytest.raises(ValueError, match="increasing order"):
        windows.update(5, [make_person(1)])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_size": 1},
        {"max_gap": -1},
        {"max_gap": 10, "stale_after": 5},
    ],
)
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TrackWindows(**kwargs)


# Normalization


def test_normalization_keeps_the_drop():
    # Standing, then the box falls and becomes short and wide.
    keypoints = np.zeros((2, 12, 3), np.float32)
    keypoints[0, :, 1] = 150.0
    keypoints[1, :, 1] = 280.0
    boxes = np.array([[80, 100, 120, 300], [0, 260, 200, 300]], np.float32)

    normalized, box_features = normalize_window(keypoints, boxes)

    # Both frames use the standing height (200 px), so the 130 px drop
    # survives as 0.65 instead of being normalized away.
    drop = normalized[1, :, 1] - normalized[0, :, 1]
    np.testing.assert_allclose(drop, 0.65, **TOL)
    np.testing.assert_allclose(box_features, [[1.0, 0.2], [0.2, 5.0]], **TOL)


def test_normalization_ignores_position_and_distance():
    keypoints = np.zeros((1, 12, 3), np.float32)
    keypoints[0, :, :2] = [110.0, 120.0]
    near = normalize_window(keypoints, np.array([[80, 100, 120, 300]], np.float32))

    # Same pose, half the size, elsewhere in the image.
    far_keypoints = keypoints.copy()
    far_keypoints[0, :, :2] = [1005.0, 510.0]
    far = normalize_window(far_keypoints, np.array([[990, 500, 1010, 600]], np.float32))

    np.testing.assert_allclose(near[0], far[0], **TOL)
    np.testing.assert_allclose(near[1], far[1], **TOL)


def test_normalization_leaves_confidence_and_input_untouched():
    rng = np.random.default_rng(0)
    keypoints = rng.uniform(0, 500, (3, 12, 3)).astype(np.float32)
    keypoints[..., 2] = rng.uniform(0, 1, (3, 12))
    original = keypoints.copy()
    boxes = np.array([[0, 0, 50, 100]] * 3, np.float32)

    normalized, _ = normalize_window(keypoints, boxes)

    np.testing.assert_array_equal(normalized[..., 2], original[..., 2])
    np.testing.assert_array_equal(keypoints, original)


def test_zero_size_box_does_not_produce_nan():
    keypoints = np.ones((2, 12, 3), np.float32)
    boxes = np.array([[5, 5, 5, 5], [5, 5, 5, 5]], np.float32)

    normalized, box_features = normalize_window(keypoints, boxes)

    assert np.isfinite(normalized).all()
    assert np.isfinite(box_features).all()


# Real pose output


@pytest.mark.skipif(not SAMPLE_JSON.is_file(), reason="sample pose output not present")
def test_replay_of_real_pose_output():
    from temporal.pose_io import load_pose_json

    data = load_pose_json(SAMPLE_JSON)
    windows = TrackWindows()
    checked = 0

    for frame in data["frames"]:
        people = {p["track_id"]: p for p in frame["people"]}
        for track_id in windows.update(frame["frame_index"], frame["people"]):
            window = windows.window(track_id)
            assert window.flatten().shape == (30, 36)
            assert np.isfinite(window.keypoints).all()
            assert np.isfinite(window.box_features).all()
            # Padding only ever sits at the start.
            real = int(window.mask.sum())
            assert window.mask[-real:].all() and not window.mask[:-real].any()
            # The newest frame is this frame's detection, confidences untouched.
            expected = [p["confidence"] for p in people[track_id]["keypoints"]]
            np.testing.assert_allclose(window.keypoints[-1, :, 2], expected, **TOL)
            checked += 1

        # Every remaining track was seen within the last stale_after frames.
        for track_id in list(windows._tracks):
            assert frame["frame_index"] - windows._tracks[track_id].last_frame <= 30

    assert checked > 0
