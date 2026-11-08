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

from .features import SIGNAL_INDEX, SIGNAL_SLICE, extract_features
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


class TemporalFallDetector:
    def __init__(
        self,
        checkpoint,
        *,
        threshold=None,
        debounce=5,
        cooldown_seconds=30.0,
        target_fps=None,
        device="cpu",
    ):
        self.model, self.meta = load_checkpoint(checkpoint, device)
        self.checkpoint = str(checkpoint)
        self.target_fps = float(target_fps or self.meta["fps"])
        self.window_size = int(self.meta["window_size"])
        self.threshold = float(self.meta.get("threshold", 0.8) if threshold is None else threshold)
        self.windows = TrackWindows(window_size=self.window_size)
        self.gate = ThresholdGate(self.threshold, debounce, cooldown_seconds)
        self.sampler = TimeSampler(self.target_fps)
        self.probabilities = {}  # latest probability per track, for overlays
        self._frame_index = -1

    def update(self, people, timestamp):
        """Feed one frame of tracked people; return any new falls."""
        if not self.sampler.keep(timestamp):
            return []
        self._frame_index += 1
        for old_id, new_id in handoff_lost_tracks(self.windows, people, self._frame_index):
            # The person is the same, so their debounce and cooldown carry over
            # too; otherwise one fall fires once per tracker ID.
            self.gate.rename(old_id, new_id)
            if old_id in self.probabilities:
                self.probabilities[new_id] = self.probabilities.pop(old_id)
        updated = self.windows.update(self._frame_index, people)
        scored = score_tracks(self.model, self.windows, updated, self.sampler.fps)

        detections = []
        for track_id, (probability, signals) in scored.items():
            self.probabilities[track_id] = probability
            if self.gate.update(track_id, probability, timestamp):
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
