"""Augmentation, ambiguous-window margin, event scoring and cross-validation folds."""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from temporal import features as F  # noqa: E402
from temporal.augment import MIRROR_ORDER, AugmentConfig, augment_clip  # noqa: E402
from temporal.crossval import choose_threshold, make_folds  # noqa: E402
from temporal.dataset import Clip, fall_overlap, samples_from_clip, synthetic_clips, windows_from_frames  # noqa: E402
from temporal.evaluate import Alert, ClipScores, clip_outcome  # noqa: E402
from temporal.train import train_model  # noqa: E402

FPS = 12.0


def person(track_id, hip_y=300.0, x=400.0):
    keypoints = [{"x": x + j, "y": hip_y - 10.0 * j, "confidence": 0.9} for j in range(17)]
    return {"track_id": track_id, "bbox": [x - 50, hip_y - 200, x + 50, hip_y + 200], "confidence": 0.9,
            "keypoints": keypoints}


def clip(name="c", group=None, frames=60, falls=(), track_ids=None):
    return Clip(name=name, group=group or name, fps=FPS, frames=[[person(1)] for _ in range(frames)],
                times=[i / FPS for i in range(frames)], fall_intervals=list(falls), track_ids=track_ids)


# Ambiguous-window margin


def test_fall_overlap_is_the_share_of_the_fall_inside_the_window():
    assert fall_overlap(39, 30, [(40, 49)]) == 0.0
    assert fall_overlap(44, 30, [(40, 49)]) == pytest.approx(0.5)
    assert fall_overlap(60, 30, [(40, 49)]) == 1.0


def test_margin_drops_windows_with_part_of_a_fall_but_keeps_the_rest():
    frames = [[person(1)] for _ in range(100)]
    everything = windows_from_frames(frames, FPS, [(50, 59)], stride=1)
    margin = windows_from_frames(frames, FPS, [(50, 59)], stride=1, ignore_partial=True)
    partial = [s for s in everything if 0 < fall_overlap(s.end_frame, 30, [(50, 59)]) <= 0.5]
    assert partial and all(s.label == 0 for s in partial)
    assert len(margin) == len(everything) - len(partial)
    assert sum(s.label for s in margin) == sum(s.label for s in everything)


# Augmentation


def test_mirror_swaps_left_and_right_joints():
    assert MIRROR_ORDER[F.LEFT_HIP] == F.RIGHT_HIP and MIRROR_ORDER[F.NOSE] == F.NOSE
    config = AugmentConfig(mirror=1.0, rotation_degrees=0, stretch=(1, 1), jitter=0, joint_dropout=0, speed=(1, 1))
    original = clip()
    mirrored = augment_clip(original, np.random.default_rng(0), config)
    left = original.frames[0][0]["keypoints"][F.LEFT_HIP]
    assert mirrored.frames[0][0]["keypoints"][F.RIGHT_HIP]["x"] == pytest.approx(-left["x"])
    x1, _, x2, _ = mirrored.frames[0][0]["bbox"]
    assert x1 < x2


def test_speed_change_moves_the_fall_with_the_frames():
    config = AugmentConfig(mirror=0, rotation_degrees=0, stretch=(1, 1), jitter=0, joint_dropout=0, speed=(1.25, 1.25))
    fast = augment_clip(clip(frames=100, falls=[(50, 59)]), np.random.default_rng(0), config)
    assert len(fast.frames) == 80
    assert fast.fall_intervals == [(40, 47)]
    assert len(fast.times) == 80 and fast.fps == FPS


def test_augmentation_keeps_track_ids_and_the_original_clip():
    original = clip(frames=40)
    before = original.frames[5][0]["keypoints"][3]["x"]
    augmented = augment_clip(original, np.random.default_rng(1))
    assert {p["track_id"] for f in augmented.frames for p in f} == {1}
    assert original.frames[5][0]["keypoints"][3]["x"] == before


def test_synthetic_clips_come_at_the_requested_rate():
    clips = synthetic_clips(6, FPS, seed=0, fall_fraction=0.5)
    assert all(c.fps == FPS for c in clips)
    assert any(c.has_fall for c in clips)
    assert sum(len(samples_from_clip(c, stride=5)) for c in clips) > 0


# Event scoring


def scored(falls_seconds_frames, frames=120):
    c = clip(frames=frames, falls=falls_seconds_frames, track_ids=[1])
    s = ClipScores(c, probabilities=[{} for _ in range(frames)], renames=[[] for _ in range(frames)],
                   alive=[set() for _ in range(frames)], faller=[{1} for _ in range(frames)])
    return s


def test_alert_on_the_faller_near_impact_catches_the_fall():
    s = scored([(60, 66)])  # 5.0-5.5 s
    outcome = clip_outcome(s, [Alert(6.0, 72, 1, 0.99, True)], catch_after=1.5)
    assert outcome["caught"] == 1 and outcome["false_alerts"] == 0
    assert outcome["delay"] == pytest.approx(0.5)


def test_other_people_and_early_faller_alerts_are_false_and_late_ones_are_late():
    s = scored([(60, 66)])
    alerts = [Alert(2.0, 24, 1, 0.99, True),   # faller, long before the fall
              Alert(5.4, 65, 7, 0.99, False),  # someone else
              Alert(8.0, 96, 1, 0.99, True)]   # faller, after the catch window
    outcome = clip_outcome(s, alerts, catch_after=1.5)
    assert outcome["caught"] == 0 and outcome["late"] == 1 and outcome["false_alerts"] == 2


def test_normal_clip_counts_its_whole_length_as_non_fall_time():
    s = scored([], frames=120)
    outcome = clip_outcome(s, [Alert(3.0, 36, 4, 0.99, False)])
    assert outcome["false_alerts"] == 1 and outcome["non_fall_seconds"] == pytest.approx(10.0)


def test_threshold_choice_prefers_fewest_false_alerts_then_most_catches():
    sweep = [{"threshold": t, "false_alerts": f, "caught": c}
             for t, f, c in [(0.5, 3, 20), (0.6, 1, 19), (0.7, 1, 19), (0.8, 1, 19), (0.9, 1, 15)]]
    assert choose_threshold(sweep) == 0.7  # middle of 0.6-0.8


# Folds and training


def test_folds_test_every_group_once_and_spread_the_falls():
    clips = [clip(f"f{i}", falls=[(30, 35)]) for i in range(10)] + [clip(f"n{i}") for i in range(5)]
    folds = make_folds(clips, 5, seed=0)
    groups = [g for fold in folds for g in fold]
    assert sorted(groups) == sorted(c.group for c in clips)
    assert all(sum(g.startswith("f") for g in fold) == 2 for fold in folds)
    assert all(sum(g.startswith("n") for g in fold) == 1 for fold in folds)


def test_final_training_without_validation_runs_every_epoch():
    samples = samples_from_clip(clip(frames=80, falls=[(50, 56)]), stride=2)
    model, history = train_model(samples, None, epochs=3, hidden_size=8, log=lambda *_: None)
    assert history["best_epoch"] == 3 and len(history["epochs"]) == 3
    assert "val_loss" not in history["epochs"][0]
