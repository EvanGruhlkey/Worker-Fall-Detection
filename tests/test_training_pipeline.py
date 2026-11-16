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
    keypoints = [{"x": x + j, "y": hip_y - 10.0 * j, "confidence": 0.9} for j in range(12)]
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
    assert MIRROR_ORDER[F.LEFT_HIP] == F.RIGHT_HIP
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


# Fast-drop check


def _body_window(centres_y, height=400.0, width=100.0):
    """A full 30-frame window of one person whose body centre moves through ``centres_y``."""
    from utils.sliding_window import TrackWindows

    windows = TrackWindows(window_size=len(centres_y))
    for index, y in enumerate(centres_y):
        windows.update(index, [person(1, hip_y=y)])
    return windows.window(1)


def test_a_fast_drop_scores_high_and_slow_lowering_or_standing_still_scores_low():
    standing = _body_window([300.0] * 30)
    fall = _body_window([300.0] * 20 + [300.0 + 40.0 * i for i in range(1, 7)] + [540.0] * 4)  # 240 px in 0.5 s
    slow = _body_window([300.0 + 8.0 * i for i in range(30)])  # 232 px over 2.5 s
    assert F.window_drop(standing, FPS) == pytest.approx(0.0)
    assert F.window_drop(fall, FPS) > 0.4
    assert F.window_drop(slow, FPS) < 0.2


def test_fast_drop_check_blocks_alerts_without_a_drop_and_keeps_the_rest():
    from temporal.evaluate import alerts_for

    frames = 30
    s = ClipScores(clip=clip(frames=frames), probabilities=[{1: 0.99, 2: 0.99} for _ in range(frames)],
                   renames=[[] for _ in range(frames)], alive=[{1, 2} for _ in range(frames)],
                   faller=[set() for _ in range(frames)],
                   drops=[{1: 0.4, 2: 0.05} for _ in range(frames)])
    assert {a.track_id for a in alerts_for(s, 0.9, debounce=5)} == {1, 2}
    assert {a.track_id for a in alerts_for(s, 0.9, debounce=5, min_drop=0.15)} == {1}


# Test-only clips


def test_test_only_rows_are_marked_for_testing_but_not_training(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_manifest import read_clips

    sheet = tmp_path / "labels.csv"
    sheet.write_text(
        "clip_id,folder,has_fall,fall_start_s,fall_end_s,quality,group,duration_s,batch\n"
        "slow,x,1,1.0,2.0,test_only,g,5,trial1\n", encoding="utf-8")
    clips, errors = read_clips(sheet)
    assert not errors
    assert clips["slow"]["quality"] == "test_only" and clips["slow"]["batch"] == "trial1"


# Training options


def test_mirror_mode_adds_one_mirrored_copy_per_clip():
    from temporal.crossval import RunConfig, _training_clips

    clips = [clip("a"), clip("b")]
    out = _training_clips(clips, RunConfig(augment_mode="mirror", augment=5), seed=0, fps=FPS)
    assert [c.name for c in out] == ["a", "b", "a~aug0", "b~aug0"]
    left = clips[0].frames[0][0]["keypoints"][F.LEFT_HIP]["x"]
    assert out[2].frames[0][0]["keypoints"][F.RIGHT_HIP]["x"] == pytest.approx(-left)


def _signal_sample(label, hip_height, torso_angle):
    from temporal.dataset import WindowSample

    features = np.zeros((30, F.NUM_FEATURES), np.float32)
    features[:, F.SIGNAL_SLICE.start + F.SIGNAL_INDEX["hip_height"]] = hip_height
    features[:, F.SIGNAL_SLICE.start + F.SIGNAL_INDEX["torso_angle"]] = torso_angle
    return WindowSample(features, np.ones(30, bool), label, "c", 29, 1)


def test_only_low_posture_normal_windows_get_extra_weight():
    from temporal.crossval import weight_hard_negatives

    samples = [_signal_sample(0, 0.5, 0.1),   # standing, normal
               _signal_sample(0, 0.2, 0.1),   # kneeling or crouching, normal
               _signal_sample(0, 0.45, 0.9),  # lying, normal
               _signal_sample(1, 0.1, 0.9)]   # a fall: never down-weighted or up-weighted
    assert weight_hard_negatives(samples, 3.0) == 2
    assert [s.weight for s in samples] == [1.0, 3.0, 3.0, 1.0]
    assert weight_hard_negatives(samples, 1.0) == 0


def test_label_smoothing_and_weights_still_train():
    samples = samples_from_clip(clip(frames=80, falls=[(50, 56)]), stride=2)
    for sample in samples[:5]:
        sample.weight = 3.0
    model, history = train_model(samples, None, epochs=2, hidden_size=8, label_smoothing=0.1, log=lambda *_: None)
    assert len(history["epochs"]) == 2 and np.isfinite(history["epochs"][-1]["train_loss"])


# Honest reporting


def _fake_result(seeds=(0,)):
    """Two folds: fold 1 holds a fall clip (faller scores 0.97 during the fall), fold 2 a normal
    clip where a coworker scores 0.93 for 2 s (a false alert at any threshold up to 0.93)."""
    from temporal.crossval import CVResult, RunConfig

    frames = 120
    fall = Clip(name="fall", group="g_fall", fps=FPS, frames=[[] for _ in range(frames)],
                times=[i / FPS for i in range(frames)], fall_intervals=[(60, 66)], track_ids=[1], batch="trial1")
    normal = Clip(name="normal", group="g_normal", fps=FPS, frames=[[] for _ in range(frames)],
                  times=[i / FPS for i in range(frames)], fall_intervals=[], track_ids=None, batch="trial2")
    result = CVResult(RunConfig(seeds=tuple(seeds)), FPS, [fall, normal], [["g_fall"], ["g_normal"]])
    for seed in seeds:
        result.scores[(seed, "fall")] = ClipScores(
            fall, probabilities=[{1: 0.97 if 60 <= i <= 72 else 0.01} for i in range(frames)],
            renames=[[] for _ in range(frames)], alive=[{1} for _ in range(frames)],
            faller=[{1} for _ in range(frames)], drops=[{1: 0.5} for _ in range(frames)])
        result.scores[(seed, "normal")] = ClipScores(
            normal, probabilities=[{2: 0.93 if 30 <= i <= 54 else 0.01} for i in range(frames)],
            renames=[[] for _ in range(frames)], alive=[{2} for _ in range(frames)],
            faller=[set() for _ in range(frames)], drops=[{2: 0.5} for _ in range(frames)])
    return result


def test_nested_threshold_is_chosen_without_the_tested_fold():
    from temporal.crossval import nested_summary

    nested = nested_summary(_fake_result())
    by_fold = {t["fold"]: t["threshold"] for t in nested["thresholds"]}
    # Fold 1 (the fall) is tested at a threshold picked on the normal clip only: above its 0.93.
    assert by_fold[1] > 0.93
    # Fold 2 (the normal clip) is tested at a threshold picked on the fall clip only, which sees
    # no false alerts and so lands well below 0.93: the coworker's false alert is counted.
    assert by_fold[2] < 0.93
    assert nested["caught"] == 1 and nested["falls"] == 1 and nested["false_alerts"] == 1


def test_results_split_by_batch_and_average_over_seeds():
    from temporal.crossval import outcomes_at, subset_summary

    result = _fake_result(seeds=(0, 1))
    result.threshold = 0.95
    result.outcomes = {s: outcomes_at(result, s, 0.95, clips=result.clips) for s in result.config.seeds}
    trial1 = subset_summary(result, {"fall"})
    assert trial1["caught"] == 1 and trial1["falls"] == 1 and trial1["spread_caught"] == [1, 1]
    trial2 = subset_summary(result, {"normal"})
    assert trial2["falls"] == 0 and trial2["false_alerts"] == 0


# Tracking cleanup


def test_duplicate_box_on_one_body_is_removed_but_two_overlapping_people_are_kept():
    from temporal.detector import drop_duplicate_people

    full = person(1, hip_y=300.0)  # box 350-450 x 100-500
    partial = dict(person(2, hip_y=300.0), bbox=[360, 250, 440, 500])  # same joints, smaller box inside
    assert [p["track_id"] for p in drop_duplicate_people([full, partial])] == [1]
    behind = dict(person(3, hip_y=300.0, x=420.0), bbox=[360, 250, 440, 500])  # inside, but different joints
    for k in behind["keypoints"]:
        k["y"] += 120
    assert {p["track_id"] for p in drop_duplicate_people([full, behind])} == {1, 3}


def test_a_wider_gap_lets_a_new_id_take_over_a_track_lost_for_a_second():
    from temporal.detector import handoff_lost_tracks
    from utils.sliding_window import TrackWindows

    def run(max_gap):
        windows = TrackWindows(window_size=30, max_gap=max_gap)
        for i in range(20):
            windows.update(i, [person(4)])
        renamed = []
        for i in range(20, 30):  # hidden for 10 frames (0.8 s at 12 fps)
            windows.update(i, [])
        renamed += handoff_lost_tracks(windows, [person(9, hip_y=320.0)], 30)
        return renamed

    assert run(max_gap=5) == []
    assert run(max_gap=12) == [(4, 9)]
