"""Rule-based fall detector over tracked YOLO pose output.

This is a baseline so the rest of the system (events, clips, review dashboard)
works end to end. It flags a fall when a tracked person

1. drops quickly: the hip centre falls by a large share of the person's
   standing height within ``descent_window`` seconds, and then
2. stays down: the torso is near horizontal (or the bounding box is wider
   than it is tall) for at least ``down_seconds``.

Replace or extend ``FallDetector.update`` with a trained classifier later; the
pipeline only relies on its signature and the ``FallDetection`` it returns.
"""

import math
from collections import deque
from dataclasses import dataclass, field

# COCO-17 keypoint indices.
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12


@dataclass
class FallDetection:
    track_id: int
    confidence: float
    reason: str


@dataclass
class _Sample:
    time: float
    hip_y: float
    height: float
    torso_angle: float | None
    aspect: float


@dataclass
class _TrackState:
    samples: deque = field(default_factory=deque)
    down_since: float | None = None
    last_fall_time: float | None = None
    last_seen: float = 0.0


def _midpoint(keypoints, a, b, min_confidence):
    points = [keypoints[i] for i in (a, b) if keypoints[i]["confidence"] >= min_confidence]
    if not points:
        return None
    return (
        sum(p["x"] for p in points) / len(points),
        sum(p["y"] for p in points) / len(points),
    )


class FallDetector:
    def __init__(
        self,
        descent_window=1.5,
        drop_ratio=0.3,
        down_angle=55.0,
        down_aspect=1.2,
        down_seconds=0.5,
        cooldown_seconds=10.0,
        forget_seconds=3.0,
        min_keypoint_confidence=0.3,
    ):
        self.descent_window = descent_window
        self.drop_ratio = drop_ratio
        self.down_angle = down_angle
        self.down_aspect = down_aspect
        self.down_seconds = down_seconds
        self.cooldown_seconds = cooldown_seconds
        self.forget_seconds = forget_seconds
        self.min_keypoint_confidence = min_keypoint_confidence
        self._history_seconds = descent_window + down_seconds + 1.0
        self._tracks = {}

    def _sample(self, person, timestamp):
        x1, y1, x2, y2 = person["bbox"]
        width, height = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
        keypoints = person["keypoints"]

        shoulders = _midpoint(
            keypoints, LEFT_SHOULDER, RIGHT_SHOULDER, self.min_keypoint_confidence
        )
        hips = _midpoint(keypoints, LEFT_HIP, RIGHT_HIP, self.min_keypoint_confidence)

        torso_angle = None
        if shoulders and hips:
            dx = shoulders[0] - hips[0]
            dy = hips[1] - shoulders[1]  # image y grows downward
            if dx or dy:
                # 0 degrees = upright, 90 = lying flat.
                torso_angle = math.degrees(math.atan2(abs(dx), dy))

        hip_y = hips[1] if hips else (y1 + y2) / 2
        return _Sample(timestamp, hip_y, height, torso_angle, width / height)

    def _is_down(self, sample):
        if sample.torso_angle is not None and sample.torso_angle >= self.down_angle:
            return True
        return sample.aspect >= self.down_aspect

    def update(self, people, timestamp):
        """Feed one frame of tracked people; return any new falls."""
        detections = []

        for person in people:
            track_id = person.get("track_id")
            if track_id is None:
                continue

            state = self._tracks.setdefault(track_id, _TrackState())
            state.last_seen = timestamp
            sample = self._sample(person, timestamp)
            state.samples.append(sample)
            while state.samples and timestamp - state.samples[0].time > self._history_seconds:
                state.samples.popleft()

            if not self._is_down(sample):
                state.down_since = None
                continue
            if state.down_since is None:
                state.down_since = timestamp
            if timestamp - state.down_since < self.down_seconds:
                continue
            if (
                state.last_fall_time is not None
                and timestamp - state.last_fall_time < self.cooldown_seconds
            ):
                continue

            detection = self._check_descent(track_id, state, sample)
            if detection:
                state.last_fall_time = timestamp
                detections.append(detection)

        stale = [
            track_id
            for track_id, state in self._tracks.items()
            if timestamp - state.last_seen > self.forget_seconds
        ]
        for track_id in stale:
            del self._tracks[track_id]

        return detections

    def _check_descent(self, track_id, state, current):
        window_start = state.down_since - self.descent_window
        before = [s for s in state.samples if window_start <= s.time <= state.down_since]
        if not before:
            return None

        # Highest hip position before going down, normalised by standing height.
        top = min(before, key=lambda s: s.hip_y)
        standing_height = max(s.height for s in before)
        drop = (current.hip_y - top.hip_y) / standing_height
        if drop < self.drop_ratio:
            return None

        drop_score = min(1.0, drop / (2 * self.drop_ratio))
        if current.torso_angle is not None:
            posture_score = min(1.0, current.torso_angle / 90.0)
            posture = f"torso {current.torso_angle:.0f} deg from vertical"
        else:
            posture_score = min(1.0, current.aspect / (2 * self.down_aspect))
            posture = f"box aspect {current.aspect:.2f}"

        return FallDetection(
            track_id=track_id,
            confidence=round(0.5 * drop_score + 0.5 * posture_score, 3),
            reason=(
                f"hip dropped {drop:.0%} of body height in "
                f"{state.down_since - top.time:.1f}s; {posture}"
            ),
        )
