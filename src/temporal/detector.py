"""Stages 2 + 5 + 6 + 7 behind the same interface as the rule-based detector.

``TemporalFallDetector.update(people, timestamp)`` returns ``FallDetection``
objects exactly like ``backend.fall_detector.FallDetector``, so the backend
pipeline, events, clips and dashboard need no change to use the classifier.
Set ``FALL_CLASSIFIER_PATH`` in ``.env`` to switch.

``score_tracks`` and ``handoff_lost_tracks`` are shared with the offline
``fall_pipeline.py`` so both paths behave the same.
"""

import math

import numpy as np
import torch

from backend.fall_detector import FallDetection
from utils.sliding_window import TrackWindows

from .features import SIGNAL_INDEX, SIGNAL_SLICE, extract_features, window_drop
from .gate import ThresholdGate
from .model import load_checkpoint
from .sampler import TimeSampler


def handoff_lost_tracks(windows, people, frame_index):
    """Give a new track ID the window of one that just vanished next to it.

    ByteTrack often re-identifies a person mid-fall (their box turns from
    tall to wide). A new ID appearing within ``windows.max_gap`` frames of a
    lost one, within about one body height, takes over that window. Returns
    the (old_id, new_id) pairs renamed.
    """
    present = {p["track_id"] for p in people}
    renamed = []
    for person in people:
        new_id = person["track_id"]
        if new_id is None or new_id in windows:
            continue
        x1, y1, x2, y2 = person["bbox"]
        center, height = ((x1 + x2) / 2, (y1 + y2) / 2), max(y2 - y1, 1e-6)
        best, best_distance = None, None
        for old_id in windows.track_ids():
            if old_id in present:
                continue
            gap = frame_index - windows.last_seen(old_id) - 1
            if gap < 0 or gap > windows.max_gap:
                continue
            ox1, oy1, ox2, oy2 = windows.last_box(old_id)
            distance = math.dist(center, ((ox1 + ox2) / 2, (oy1 + oy2) / 2))
            if distance <= max(height, oy2 - oy1) and (best_distance is None or distance < best_distance):
                best, best_distance = old_id, distance
        if best is not None:
            windows.rename(best, new_id)
            present.add(new_id)
            renamed.append((best, new_id))
    return renamed


DUPLICATE_CONTAINMENT = 0.9  # share of the smaller box inside the bigger one
DUPLICATE_POSE_DISTANCE = 0.08  # median joint distance, in heights of the bigger box


def drop_duplicate_people(people):
    """Remove the second detection of a body that is already in the frame.

    The detector sometimes puts a second, smaller box on one worker (around
    their legs, say) and the tracker gives it its own ID, so the same body is
    scored twice, once from a partial box. A box counts as a duplicate when at
    least 90% of it lies inside a bigger box *and* their joints are in nearly
    the same places (median distance under 8% of the bigger box's height):
    two real people overlapping (one behind the other) have different joints
    and are both kept. The bigger box is kept. Across Trial 1's rig and
    CAUCAFall clips this never removed a faller.
    """
    if len(people) < 2:
        return people

    def area(box):
        return max(box[2] - box[0], 0) * max(box[3] - box[1], 0)

    def inside(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / max(area(a), 1e-6)

    def same_joints(p, q):
        a = np.array([[k["x"], k["y"], k["confidence"]] for k in p["keypoints"]], np.float32)
        b = np.array([[k["x"], k["y"], k["confidence"]] for k in q["keypoints"]], np.float32)
        seen = (a[:, 2] > 0.3) & (b[:, 2] > 0.3)
        if seen.sum() < 3:
            return False
        height = max(q["bbox"][3] - q["bbox"][1], p["bbox"][3] - p["bbox"][1], 1e-6)
        return float(np.median(np.linalg.norm(a[seen, :2] - b[seen, :2], axis=1))) / height < DUPLICATE_POSE_DISTANCE

    drop = set()
    for i, p in enumerate(people):
        for j, q in enumerate(people):
            if i == j or j in drop:
                continue
            smaller = area(p["bbox"]) < area(q["bbox"]) or (area(p["bbox"]) == area(q["bbox"]) and i > j)
            if smaller and inside(p["bbox"], q["bbox"]) >= DUPLICATE_CONTAINMENT and same_joints(p, q):
                drop.add(i)
                break
    return [p for i, p in enumerate(people) if i not in drop] if drop else people


def score_tracks(model, windows, track_ids, fps):
    """Classify every full window among ``track_ids`` in one batch.

    Returns ``{track_id: (probability, signals)}`` where ``signals`` is the
    newest frame's SIGNAL_NAMES row.
    """
    batch, ids = [], []
    for track_id in track_ids:
        window = windows.window(track_id)
        if window is None or not window.is_full:
            continue
        batch.append(extract_features(window, fps))
        ids.append(track_id)
    if not batch:
        return {}
    features = torch.from_numpy(np.stack(batch))
    probabilities = model.predict_proba(features).cpu().numpy()
    return {
        track_id: (float(p), feats[-1, SIGNAL_SLICE])
        for track_id, p, feats in zip(ids, probabilities, batch)
    }


def track_drops(windows, track_ids, fps):
    """Fastest body drop (``features.window_drop``) of every full window among ``track_ids``."""
    drops = {}
    for track_id in track_ids:
        window = windows.window(track_id)
        if window is not None and window.is_full:
            drops[track_id] = window_drop(window, fps)
    return drops


def drop_checked(probabilities, drops, min_drop):
    """The fast-drop check: a track whose body has not dropped at least
    ``min_drop`` body lengths within the window cannot raise an alert, however
    sure the model is (it is lying, kneeling or crouching, not falling)."""
    if min_drop <= 0:
        return probabilities
    return {t: (p if drops.get(t, 0.0) >= min_drop else 0.0) for t, p in probabilities.items()}


class TemporalFallDetector:
    def __init__(
        self,
        checkpoint,
        *,
        threshold=None,
        debounce=None,
        cooldown_seconds=30.0,
        target_fps=None,
        min_drop=None,
        device="cpu",
    ):
        # Track gap / handoff and duplicate removal come from the checkpoint,
        # so live detection tracks people exactly as training did.
        self.model, self.meta = load_checkpoint(checkpoint, device)
        self.checkpoint = str(checkpoint)
        self.target_fps = float(target_fps or self.meta["fps"])
        self.window_size = int(self.meta["window_size"])
        self.threshold = float(self.meta.get("threshold", 0.8) if threshold is None else threshold)
        self.min_drop = float(self.meta.get("min_drop", 0.0) if min_drop is None else min_drop)
        self.max_gap = int(self.meta.get("max_gap", 5))
        self.merge_duplicates = bool(self.meta.get("merge_duplicates", False))
        self.windows = TrackWindows(window_size=self.window_size, max_gap=self.max_gap)
        # Frames above the threshold before an alert: from the checkpoint (5 for older ones).
        debounce = int(self.meta.get("debounce", 5) if debounce is None else debounce)
        self.gate = ThresholdGate(self.threshold, debounce, cooldown_seconds)
        self.sampler = TimeSampler(self.target_fps)
        self.probabilities = {}  # latest probability per track, for overlays
        self._frame_index = -1

    def update(self, people, timestamp):
        """Feed one frame of tracked people; return any new falls."""
        if not self.sampler.keep(timestamp):
            return []
        self._frame_index += 1
        if self.merge_duplicates:
            people = drop_duplicate_people(people)
        for old_id, new_id in handoff_lost_tracks(self.windows, people, self._frame_index):
            # The person is the same, so their debounce and cooldown carry over
            # too; otherwise one fall fires once per tracker ID.
            self.gate.rename(old_id, new_id)
            if old_id in self.probabilities:
                self.probabilities[new_id] = self.probabilities.pop(old_id)
        updated = self.windows.update(self._frame_index, people)
        scored = score_tracks(self.model, self.windows, updated, self.sampler.fps)
        drops = track_drops(self.windows, scored, self.sampler.fps) if self.min_drop > 0 else {}
        checked = drop_checked({t: p for t, (p, _) in scored.items()}, drops, self.min_drop)

        detections = []
        for track_id, (probability, signals) in scored.items():
            self.probabilities[track_id] = probability
            if self.gate.update(track_id, checked[track_id], timestamp):
                detections.append(
                    FallDetection(
                        track_id=track_id,
                        confidence=round(probability, 3),
                        reason=(
                            f"classifier p={probability:.2f} for {self.gate.debounce} frames; "
                            f"hip velocity {signals[SIGNAL_INDEX['hip_velocity']]:+.1f} heights/s; "
                            f"torso {signals[SIGNAL_INDEX['torso_angle']] * 90:.0f} deg from vertical"
                        ),
                    )
                )
        for track_id in list(self.probabilities):
            if track_id not in self.windows:
                del self.probabilities[track_id]
                self.gate.forget(track_id)
        return detections
