import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sliding_window import TrackWindows  # noqa: E402
from temporal import features as F  # noqa: E402
from temporal.dataset import (  # noqa: E402
    label_for_window,
    samples_from_sequences,
    split_by_source,
    stack,
    windows_from_frames,
)
from temporal.detector import TemporalFallDetector, handoff_lost_tracks  # noqa: E402
from temporal.gate import ThresholdGate  # noqa: E402
from temporal.model import FallGRU, FallTCN, build_model, load_checkpoint, save_checkpoint  # noqa: E402
from temporal.sampler import FrameSampler, TimeSampler  # noqa: E402
from temporal.synthetic import ACTIVITIES, generate_dataset, generate_sequence  # noqa: E402
from temporal.train import evaluate, train_model  # noqa: E402
from temporal.visualize import SignalPanel, TrackSignals, compose, plot_track_timeline  # noqa: E402

BOOTSTRAP_CHECKPOINT = REPO_ROOT / "models" / "fall_gru.pt"
FPS = 15.0


def person(track_id, hip_y, *, x=400.0, height=400.0, lying=False, conf=0.9):
    """An upright (or lying) stick figure with shoulders, hips and nose placed."""
    keypoints = [{"x": x, "y": hip_y, "confidence": conf} for _ in range(17)]
    if lying:
        shoulders, nose = (x + 150.0, hip_y), (x + 220.0, hip_y)
        bbox = [x - 200.0, hip_y - 50.0, x + 250.0, hip_y + 50.0]
    else:
        shoulders, nose = (x, hip_y - 150.0), (x, hip_y - 200.0)
        bbox = [x - 50.0, hip_y - height / 2, x + 50.0, hip_y + height / 2]
    keypoints[F.NOSE] = {"x": nose[0], "y": nose[1], "confidence": conf}
    for index in (F.LEFT_SHOULDER, F.RIGHT_SHOULDER):
        keypoints[index] = {"x": shoulders[0], "y": shoulders[1], "confidence": conf}
    for index in (F.LEFT_HIP, F.RIGHT_HIP):
        keypoints[index] = {"x": x, "y": hip_y, "confidence": conf}
    return {"track_id": track_id, "bbox": bbox, "confidence": 0.9, "keypoints": keypoints}


def window_from(frames, window_size=30):
    windows = TrackWindows(window_size=window_size)
    for index, people in enumerate(frames):
        windows.update(index, people)
    return windows.window(1)


# Features


def test_feature_vector_has_documented_layout():
    window = window_from([[person(1, 300.0)] for _ in range(30)])
    feats = F.extract_features(window, FPS)
    assert feats.shape == (30, F.NUM_FEATURES)
    assert feats.dtype == np.float32
    assert len(F.feature_names()) == F.NUM_FEATURES
    assert F.feature_names()[F.SIGNAL_SLICE] == list(F.SIGNAL_NAMES)
    # A person standing still has no velocity or acceleration anywhere.
    assert not feats[:, 51:119].any()


def test_padded_frames_are_zero_and_no_velocity_crosses_the_padding():
    window = window_from([[person(1, 300.0 + 10 * i)] for i in range(5)])
    feats = F.extract_features(window, FPS)
    assert not feats[:25].any()
    # First real frame has no previous real frame, so its velocity is zero.
    assert not feats[25, 51:85].any()
    assert feats[26, 51:85].any()


def test_velocity_is_in_body_heights_per_second():
    # Hips drop 10 px per frame; standing height 400 px; at 15 fps that is
    # 10 / 400 * 15 = 0.375 heights per second, downward.
    window = window_from([[person(1, 300.0 + 10 * i)] for i in range(30)])
    feats = F.extract_features(window, FPS)
    hip_y_velocity = feats[-1, 51 + F.LEFT_HIP * 2 + 1]
    np.testing.assert_allclose(hip_y_velocity, 0.375, atol=1e-3)
    signal = feats[-1, F.SIGNAL_SLICE][F.SIGNAL_INDEX["hip_velocity"]]
    np.testing.assert_allclose(signal, -0.375, atol=1e-3)  # upward positive


def test_signals_distinguish_upright_from_lying():
    standing = [[person(1, 300.0)] for _ in range(5)]
    # Heights are relative to the tallest box in the window (standing height),
    # so the window must contain the upright frames for "lying" to read low.
    fallen = standing + [[person(1, 500.0, lying=True)] for _ in range(5)]
    upright = F.window_signals(window_from(standing), FPS)
    lying = F.window_signals(window_from(fallen), FPS)
    assert upright[F.SIGNAL_INDEX["torso_angle"]] < 0.05
    assert lying[F.SIGNAL_INDEX["torso_angle"]] > 0.95
    assert upright[F.SIGNAL_INDEX["hip_height"]] > 0.4
    assert lying[F.SIGNAL_INDEX["hip_height"]] < 0.2
    assert lying[F.SIGNAL_INDEX["head_height"]] < 0.2


def test_missing_joint_is_carried_forward_not_jumped():
    frames = [[person(1, 300.0)] for _ in range(6)]
    # Frame 3: the nose is lost; YOLO reports junk far away with low confidence.
    frames[3][0]["keypoints"][F.NOSE] = {"x": 9000.0, "y": 9000.0, "confidence": 0.1}
    feats = F.extract_features(window_from(frames, window_size=6), FPS)
    nose_x = feats[:, F.NOSE * 3]
    np.testing.assert_allclose(nose_x, nose_x[0], atol=1e-5)
    assert not feats[:, 51:85].any()  # no velocity spike from the junk point
    assert feats[3, F.NOSE * 3 + 2] == pytest.approx(0.1)  # confidence is kept


# Model and checkpoints


@pytest.mark.parametrize("arch", ["gru", "tcn"])
def test_models_are_small_and_output_one_probability_per_window(arch):
    model = build_model(arch)
    assert 10_000 <= model.count_parameters() <= 300_000
    x = torch.randn(3, 30, F.NUM_FEATURES)
    mask = torch.ones(3, 30, dtype=torch.bool)
    probabilities = model.predict_proba(x, mask)
    assert probabilities.shape == (3,)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_tcn_sees_the_whole_window():
    assert FallTCN().receptive_field >= 30


def test_checkpoint_round_trip(tmp_path):
    model = FallGRU(hidden_size=16)
    model.set_feature_scaling(np.full(F.NUM_FEATURES, 0.5), np.full(F.NUM_FEATURES, 2.0))
    x = torch.randn(2, 30, F.NUM_FEATURES)
    expected = model.predict_proba(x)

    path = tmp_path / "model.pt"
    save_checkpoint(path, model, fps=15, window_size=30, threshold=0.7, metrics={"f1": 0.9})
    loaded, meta = load_checkpoint(path)

    torch.testing.assert_close(loaded.predict_proba(x), expected)
    assert meta["arch"] == "gru" and meta["config"]["hidden_size"] == 16
    assert meta["fps"] == 15 and meta["window_size"] == 30 and meta["threshold"] == 0.7
    assert meta["feature_version"] == F.FEATURE_VERSION


def test_checkpoint_from_other_feature_version_is_refused(tmp_path):
    path = tmp_path / "model.pt"
    save_checkpoint(path, FallGRU(hidden_size=8), fps=15, window_size=30)
    data = torch.load(path, weights_only=True)
    data["feature_version"] = F.FEATURE_VERSION + 1
    torch.save(data, path)
    with pytest.raises(ValueError, match="feature version"):
        load_checkpoint(path)


# Threshold gate


def test_gate_needs_consecutive_frames_then_fires_once():
    gate = ThresholdGate(threshold=0.8, debounce=3, cooldown_seconds=10.0)
    assert [gate.update(1, p, t / FPS) for t, p in enumerate([0.9, 0.9, 0.5, 0.9, 0.9])] == [False] * 5
    assert gate.update(1, 0.95, 5 / FPS)  # third consecutive frame above
    assert not gate.update(1, 0.95, 6 / FPS)  # cooldown
    assert gate.update(1, 0.95, 11.0)  # cooldown over, still above


def test_gate_keeps_tracks_apart():
    gate = ThresholdGate(threshold=0.8, debounce=2)
    gate.update(1, 0.9, 0.0)
    assert not gate.update(2, 0.9, 0.0)
    assert gate.update(1, 0.9, 0.1)


@pytest.mark.parametrize("kwargs", [{"threshold": 1.5}, {"debounce": 0}, {"cooldown_seconds": -1}])
def test_gate_rejects_bad_settings(kwargs):
    with pytest.raises(ValueError):
        ThresholdGate(**kwargs)


# Frame sampler


@pytest.mark.parametrize("source, stride, fps", [(30, 2, 15), (25, 2, 12.5), (15, 1, 15), (60, 4, 15), (10, 1, 10)])
def test_frame_sampler_stride(source, stride, fps):
    sampler = FrameSampler(source, 15)
    assert sampler.stride == stride and sampler.fps == fps
    kept = [i for i in range(10) if sampler.keep(i)]
    assert kept == list(range(0, 10, stride))
    assert [sampler.processed_index(i) for i in kept] == list(range(len(kept)))


def test_time_sampler_halves_a_30fps_stream():
    sampler = TimeSampler(15)
    kept = [i for i in range(30) if sampler.keep(i / 30)]
    assert kept == list(range(0, 30, 2))
    assert sampler.fps == pytest.approx(15, rel=0.05)


# Synthetic data and dataset


@pytest.mark.parametrize("activity", ACTIVITIES)
def test_synthetic_sequences_are_valid_pose_output(activity):
    sequence = generate_sequence(np.random.default_rng(3), activity, 15)
    assert sequence.fps == 15
    assert len(sequence.frames) >= 30
    for people in sequence.frames:
        for p in people:
            assert len(p["keypoints"]) == 17
            x1, y1, x2, y2 = p["bbox"]
            assert x2 > x1 and y2 > y1
            assert np.isfinite([[k["x"], k["y"], k["confidence"]] for k in p["keypoints"]]).all()
    if activity.startswith("fall"):
        first, last = sequence.fall_interval
        assert 0 < first < last < len(sequence.frames)
    else:
        assert sequence.fall_interval is None


def test_window_label_needs_more_than_half_the_fall():
    fall = [(100, 109)]  # ten frames
    assert label_for_window(end_frame=104, window_size=30, fall_intervals=fall) == 0  # 5 of 10
    assert label_for_window(end_frame=106, window_size=30, fall_intervals=fall) == 1  # 7 of 10
    assert label_for_window(end_frame=129, window_size=30, fall_intervals=fall) == 1  # all inside
    assert label_for_window(end_frame=135, window_size=30, fall_intervals=fall) == 0  # slid out


def test_windows_follow_the_window_rules():
    frames = [[person(1, 300.0)] for _ in range(60)]
    samples = windows_from_frames(frames, FPS, [(40, 45)], source="v", stride=5)
    ends = [s.end_frame for s in samples]
    assert ends == [30, 35, 40, 45, 50, 55]  # full windows only, every 5th frame
    assert [s.label for s in samples] == [0, 0, 0, 1, 1, 1]


def test_split_keeps_whole_sources_together():
    sequences = generate_dataset(40, seed=1)
    samples = samples_from_sequences(sequences)
    train, val, test = split_by_source(samples, seed=1)
    sources = [{s.source for s in part} for part in (train, val, test)]
    assert not (sources[0] & sources[1]) and not (sources[0] & sources[2]) and not (sources[1] & sources[2])
    assert len(train) + len(val) + len(test) == len(samples)


def test_training_learns_synthetic_falls():
    sequences = generate_dataset(80, seed=2)
    train, val, test = split_by_source(samples_from_sequences(sequences), seed=2)
    model, history = train_model(train, val, arch="gru", epochs=8, hidden_size=32, log=lambda *_: None)
    assert history["epochs"][-1]["train_loss"] < history["epochs"][0]["train_loss"]
    result = evaluate(model, test, thresholds=(0.5,))["at"][0]
    assert result["f1"] > 0.5


# Track handoff


def test_new_id_near_a_lost_track_inherits_its_window():
    windows = TrackWindows(window_size=30)
    for index in range(20):
        windows.update(index, [person(1, 300.0)])
    # Track 1 vanishes; track 2 appears one frame later at the same spot.
    windows.update(20, [])
    people = [person(2, 310.0)]
    assert handoff_lost_tracks(windows, people, 21) == [(1, 2)]
    windows.update(21, people)
    assert 2 in windows and 1 not in windows
    assert windows.window(2).mask.sum() == 22  # 20 frames + 1 filled gap + 1 new


def test_far_away_new_id_is_not_merged():
    windows = TrackWindows(window_size=30)
    windows.update(0, [person(1, 300.0, x=100.0)])
    windows.update(1, [])
    assert handoff_lost_tracks(windows, [person(2, 300.0, x=1500.0)], 2) == []


# Visualisation


def test_signal_panel_renders_an_image_of_the_requested_size(tmp_path):
    history = TrackSignals(7)
    for index in range(40):
        signals = np.array([0.5, 0.9, 0.1, -0.2, 0.1], np.float32)
        history.append(index / FPS, index, signals, 1.0, probability=0.3, above=False)
    history.events.append(1.0)
    panel = SignalPanel(width=320, height=360)
    image = panel.render(history, now=40 / FPS, label="Track 7")
    assert image.shape == (360, 320, 3) and image.dtype == np.uint8
    assert panel.render(None, now=0.0).shape == (360, 320, 3)

    frame = np.zeros((720, 1280, 3), np.uint8)
    assert compose(frame, image).shape == (360, 640 + 320, 3)

    path = plot_track_timeline(history, tmp_path / "track_7.png")
    assert Path(path).stat().st_size > 1000
    assert history.to_dict()["events"] == [1.0]


# The bootstrap checkpoint, end to end through the drop-in detector


@pytest.mark.skipif(not BOOTSTRAP_CHECKPOINT.is_file(), reason="bootstrap checkpoint not trained")
def test_bootstrap_detector_flags_a_synthetic_fall_and_not_a_crouch():
    detector = TemporalFallDetector(BOOTSTRAP_CHECKPOINT, cooldown_seconds=30.0)
    rng = np.random.default_rng(7)

    fall = generate_sequence(rng, "fall", 15)
    detections = []
    for index, people in enumerate(fall.frames):
        detections += detector.update(people, index / 15)
    assert len(detections) == 1
    assert detections[0].track_id == 1 and detections[0].confidence >= detector.threshold
    assert "classifier" in detections[0].reason

    for activity in ("crouch", "walk", "lie_still"):
        detector = TemporalFallDetector(BOOTSTRAP_CHECKPOINT)
        sequence = generate_sequence(rng, activity, 15)
        assert not any(detector.update(people, i / 15) for i, people in enumerate(sequence.frames)), activity


@pytest.mark.skipif(not BOOTSTRAP_CHECKPOINT.is_file(), reason="bootstrap checkpoint not trained")
def test_tracker_id_switch_mid_fall_still_fires_exactly_once():
    fall = generate_sequence(np.random.default_rng(11), "fall", 15)
    first, last = fall.fall_interval
    switch = (first + last) // 2
    for index, people in enumerate(fall.frames):
        for p in people:
            p["track_id"] = 2 if index >= switch else 1
    detector = TemporalFallDetector(BOOTSTRAP_CHECKPOINT)
    detections = []
    for index, people in enumerate(fall.frames):
        detections += detector.update(people, index / 15)
    assert [d.track_id for d in detections] == [2]
