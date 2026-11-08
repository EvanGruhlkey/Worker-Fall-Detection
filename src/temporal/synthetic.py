"""Procedural skeleton sequences for bootstrapping the classifier.

There is no labeled site footage yet, so this module animates a 17-joint stick
figure through everyday activities (standing, walking, crouching, bending,
sitting, lying) and through falls, and outputs the 12 body joints as the ``people`` lists
that ``yolo_pose.process_frame`` would produce. The sequences then run through
the real stage 5 window and feature code, so a model trained on them is wired
up exactly like one trained on real footage.

This is not a substitute for real data: the figures are 2-D, their motion is
idealised, and the camera never moves. It exists so stages 6 and 7 can be
built, tested and visualised end to end before the labeled dataset exists.
The guide's domain-gap warning applies in full.

Coordinates inside ``skeleton`` are body heights with the ground at y = 0 and y
growing downward (image convention), so a standing head is near y = -0.95.
"""

import math
from dataclasses import dataclass

import numpy as np

FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
FRAME_RATES = (10.0, 12.5, 15.0)

ACTIVITIES = (
    "stand",
    "walk",
    "crouch",
    "bend",
    "sit",
    "lie_still",
    "lie_down_slowly",
    "trip_recover",
    "fall",
    "fall_toward_camera",
)
FALL_ACTIVITIES = frozenset({"fall", "fall_toward_camera"})

# Segment lengths in body heights.
TORSO, NECK, HEAD = 0.30, 0.06, 0.07
SHOULDER_HALF_WIDTH, HIP_HALF_WIDTH = 0.12, 0.08
UPPER_ARM, FOREARM = 0.17, 0.17
THIGH, SHIN = 0.245, 0.245
STANDING_HIP = 0.52

# Pose parameters: torso angle from vertical in degrees (signed: which way the
# shoulders lean), hip height above the ground, horizontal position, arm swing
# angle in radians, and "spread", the horizontal foreshortening of the torso
# (1 = seen from the side, small = falling toward the camera).
THETA, HIP, X, ARM, SPREAD = range(5)


@dataclass
class Sequence:
    activity: str
    fps: float
    frames: list  # one `people` list per frame, as yolo_pose.process_frame returns
    fall_interval: tuple | None  # (first_frame, last_frame) of the fall, inclusive

    @property
    def is_fall(self):
        return self.fall_interval is not None


def _rotate(vector, angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]])


def _unit(vector):
    norm = float(np.hypot(*vector))
    return vector / norm if norm > 1e-9 else np.array([0.0, 1.0])


def skeleton(theta, hip, x, arm, spread=1.0, facing=1.0):
    """Return (17, 2) COCO joints in body-height units for one pose."""
    angle = math.radians(theta)
    up = np.array([math.sin(angle) * spread, -math.cos(angle)])
    perp = np.array([math.cos(angle), math.sin(angle) * spread])
    down = _unit(-up)

    pelvis = np.array([x, -hip])
    shoulder_center = pelvis + TORSO * up
    neck = shoulder_center + NECK * up
    nose = neck + HEAD * up + 0.02 * facing * perp
    left_eye, right_eye = nose + 0.02 * up + 0.03 * perp, nose + 0.02 * up - 0.03 * perp
    left_ear, right_ear = nose + 0.01 * up + 0.06 * perp, nose + 0.01 * up - 0.06 * perp
    left_shoulder = shoulder_center + SHOULDER_HALF_WIDTH * perp
    right_shoulder = shoulder_center - SHOULDER_HALF_WIDTH * perp
    left_hip, right_hip = pelvis + HIP_HALF_WIDTH * perp, pelvis - HIP_HALF_WIDTH * perp

    left_elbow = left_shoulder + UPPER_ARM * _rotate(down, arm)
    left_wrist = left_elbow + FOREARM * _rotate(down, 1.3 * arm)
    right_elbow = right_shoulder + UPPER_ARM * _rotate(down, -arm)
    right_wrist = right_elbow + FOREARM * _rotate(down, -1.3 * arm)

    # Legs hang straight down when upright and extend along the body axis when
    # lying; in between they reach the ground, bending at the knee if the hips
    # are too low for straight legs.
    lean = min(1.0, abs(theta) / 90.0)
    axis = _unit((1.0 - lean) * np.array([0.0, 1.0]) + lean * down)
    forward = np.array([axis[1], -axis[0]]) * facing
    legs = []
    for hip_point in (left_hip, right_hip):
        if axis[1] > 0.05:
            reach = -hip_point[1] / axis[1]
            length = min(THIGH + SHIN, max(reach, 0.15))
        else:
            length = THIGH + SHIN
        ankle = hip_point + length * axis
        bend = math.sqrt(max(0.0, THIGH**2 - (length / 2) ** 2))
        knee = hip_point + (length / 2) * axis + bend * forward
        legs.append((knee, ankle))

    return np.array(
        [
            nose, left_eye, right_eye, left_ear, right_ear,
            left_shoulder, right_shoulder, left_elbow, right_elbow,
            left_wrist, right_wrist, left_hip, right_hip,
            legs[0][0], legs[1][0], legs[0][1], legs[1][1],
        ]
    )


# Easing functions map segment progress 0..1 to 0..1.
def _linear(u):
    return u


def _smooth(u):
    return u * u * (3.0 - 2.0 * u)


def _accelerate(u):
    """Speed up over the segment, like a body giving way to gravity."""
    return u * u


def _interpolate(keyframes, times):
    """Sample keyframes [(time, params, easing), ...] at the given times."""
    params = np.zeros((len(times), 5), np.float64)
    for index, t in enumerate(times):
        if t >= keyframes[-1][0]:
            params[index] = keyframes[-1][1]
            continue
        for (t0, p0, _), (t1, p1, easing) in zip(keyframes, keyframes[1:]):
            if t0 <= t < t1:
                u = easing((t - t0) / (t1 - t0))
                params[index] = p0 + (p1 - p0) * u
                break
    return params


def _standing(rng, lean=1.0):
    return np.array([rng.normal(0.0, 3.0) * lean, rng.uniform(0.49, 0.54), 0.0, 0.0, 1.0])


def _activity_keyframes(rng, activity, direction):
    """Return (keyframes, fall interval in seconds or None, walk_until seconds)."""
    stand = _standing(rng)
    frames = [(0.0, stand, _linear)]
    fall, walk_until = None, None

    def add(duration, params, easing=_smooth):
        frames.append((frames[-1][0] + duration, np.array(params, np.float64), easing))

    def hold(duration):
        add(duration, frames[-1][1], _linear)

    if activity == "stand":
        hold(rng.uniform(2.5, 5.0))

    elif activity == "walk":
        duration = rng.uniform(2.5, 5.0)
        lean = stand.copy()
        lean[THETA] = direction * rng.uniform(3.0, 10.0)
        frames[0] = (0.0, lean, _linear)
        add(duration, [lean[THETA], lean[HIP], direction * rng.uniform(1.0, 2.5), 0.0, 1.0], _linear)
        walk_until = duration

    elif activity == "crouch":
        hold(rng.uniform(0.5, 1.5))
        add(rng.uniform(0.9, 2.5), [direction * rng.uniform(10, 40), rng.uniform(0.17, 0.30), 0.0, 0.0, 1.0])
        hold(rng.uniform(0.8, 2.0))
        if rng.random() < 0.5:
            add(rng.uniform(0.8, 2.0), stand)

    elif activity == "bend":
        hold(rng.uniform(0.5, 1.5))
        add(rng.uniform(0.7, 2.0), [direction * rng.uniform(50, 85), rng.uniform(0.42, 0.50), 0.0, 0.0, 1.0])
        hold(rng.uniform(0.8, 2.0))
        if rng.random() < 0.7:
            add(rng.uniform(0.7, 2.0), stand)

    elif activity == "sit":
        hold(rng.uniform(0.5, 1.5))
        add(rng.uniform(0.8, 2.0), [direction * rng.uniform(5, 20), rng.uniform(0.26, 0.36), 0.0, 0.0, 1.0])
        hold(rng.uniform(1.0, 3.0))

    elif activity == "lie_still":
        lying = np.array([direction * rng.uniform(78, 92), rng.uniform(0.07, 0.14), 0.0, 0.0, rng.uniform(0.6, 1.0)])
        frames[0] = (0.0, lying, _linear)
        hold(rng.uniform(2.5, 5.0))

    elif activity == "lie_down_slowly":
        hold(rng.uniform(0.3, 1.0))
        add(rng.uniform(1.0, 2.0), [direction * rng.uniform(20, 40), rng.uniform(0.2, 0.3), 0.0, 0.0, 1.0])
        add(rng.uniform(1.5, 3.0), [direction * rng.uniform(78, 90), rng.uniform(0.08, 0.13), 0.0, 0.0, 1.0])
        hold(rng.uniform(1.0, 2.5))

    elif activity == "trip_recover":
        pre = rng.uniform(0.8, 2.0)
        hold(pre)
        add(rng.uniform(0.25, 0.5), [direction * rng.uniform(20, 45), rng.uniform(0.36, 0.46), 0.0, 0.0, 1.0], _accelerate)
        add(rng.uniform(0.5, 1.0), stand)
        hold(rng.uniform(0.8, 1.5))
        if rng.random() < 0.5:
            walk_until = pre

    elif activity in FALL_ACTIVITIES:
        pre = rng.uniform(0.8, 2.5)
        hold(pre)
        duration = rng.uniform(0.3, 0.8)
        toward_camera = activity == "fall_toward_camera"
        add(
            duration,
            [
                direction * rng.uniform(70, 92),
                rng.uniform(0.06, 0.15),
                0.0,
                0.0,
                rng.uniform(0.2, 0.45) if toward_camera else 1.0,
            ],
            _accelerate,
        )
        fall = (pre, pre + duration)
        hold(rng.uniform(1.0, 3.0))
        if rng.random() < 0.5:
            walk_until = pre
    else:
        raise ValueError(f"Unknown activity {activity!r}")

    return frames, fall, walk_until


def _wobble(rng, times, amplitude):
    """Slow, smooth noise: two sinusoids with random frequency and phase."""
    total = np.zeros(len(times))
    for _ in range(2):
        frequency, phase = rng.uniform(0.2, 0.9), rng.uniform(0, 2 * math.pi)
        total += np.sin(2 * math.pi * frequency * times + phase)
    return amplitude * total / 2


def generate_sequence(rng, activity=None, fps=None):
    """Animate one activity and render it as per-frame ``people`` lists."""
    activity = activity or rng.choice(ACTIVITIES)
    fps = float(fps or rng.choice(FRAME_RATES))
    direction = float(rng.choice([-1.0, 1.0]))
    facing = direction

    keyframes, fall_seconds, walk_until = _activity_keyframes(rng, activity, direction)
    duration = keyframes[-1][0]
    times = np.arange(0.0, duration, 1.0 / fps)
    params = _interpolate(keyframes, times)

    if walk_until is not None:
        # Steps bob the hips and swing the arms until the walk stops.
        envelope = np.clip((walk_until + 0.3 - times) / 0.3, 0.0, 1.0)
        cadence = 2 * math.pi * rng.uniform(1.6, 2.1) * times
        params[:, HIP] += 0.015 * np.sin(cadence) * envelope
        params[:, ARM] += 0.5 * np.sin(cadence) * envelope
    params[:, THETA] += _wobble(rng, times, 2.0)
    params[:, HIP] += _wobble(rng, times, 0.008)
    params[:, X] += _wobble(rng, times, 0.01)

    joints = np.stack([skeleton(*p, facing=facing) for p in params])  # (frames, 17, 2)

    # Camera: pixels per body height, mirror, and where the feet are.
    scale = rng.uniform(90.0, 420.0)
    flip = float(rng.choice([-1.0, 1.0]))
    center_x = rng.uniform(0.25, 0.75) * FRAME_WIDTH
    ground_y = rng.uniform(0.6, 0.95) * FRAME_HEIGHT
    pixels = np.empty_like(joints)
    pixels[..., 0] = center_x + flip * scale * joints[..., 0]
    pixels[..., 1] = ground_y + scale * joints[..., 1]

    # Pose-estimator jitter, plus a few runs of low-confidence, badly placed joints.
    jitter = scale * rng.uniform(0.004, 0.015)
    pixels += rng.normal(0.0, jitter, pixels.shape)
    confidence = np.clip(
        rng.uniform(0.55, 0.98, 17)[None, :] + rng.normal(0.0, 0.05, pixels.shape[:2]), 0.05, 1.0
    )
    for _ in range(rng.integers(0, 4)):
        joint = rng.integers(0, 17)
        start = rng.integers(0, max(1, len(times) - 3))
        stop = min(len(times), start + rng.integers(2, 12))
        confidence[start:stop, joint] = rng.uniform(0.0, 0.25, stop - start)
        pixels[start:stop, joint] += rng.normal(0.0, 4 * jitter, (stop - start, 2))

    margin = scale * 0.03
    frames = []
    for frame_pixels, frame_confidence in zip(pixels, confidence):
        x1, y1 = frame_pixels.min(axis=0) - margin + rng.normal(0.0, 0.01 * scale, 2)
        x2, y2 = frame_pixels.max(axis=0) + margin + rng.normal(0.0, 0.01 * scale, 2)
        frames.append(
            [
                {
                    "track_id": 1,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(rng.uniform(0.6, 0.95)),
                    "keypoints": [
                        {"x": float(px), "y": float(py), "confidence": float(c)}
                        for (px, py), c in zip(frame_pixels[5:], frame_confidence[5:])
                    ],
                }
            ]
        )

    fall_interval = None
    if fall_seconds is not None:
        first = int(round(fall_seconds[0] * fps))
        last = min(len(frames) - 1, int(round(fall_seconds[1] * fps)))
        fall_interval = (first, last)

    # Short tracker drop-outs (never during the fall) exercise the gap filling.
    for _ in range(rng.integers(0, 3)):
        start = int(rng.integers(0, len(frames)))
        stop = min(len(frames), start + int(rng.integers(1, 4)))
        if fall_interval and start <= fall_interval[1] + 2 and stop >= fall_interval[0] - 2:
            continue
        for index in range(start, stop):
            frames[index] = []

    return Sequence(activity, fps, frames, fall_interval)


def generate_dataset(count, seed=0, fall_fraction=0.4):
    """Return `count` sequences; about `fall_fraction` of them contain a fall."""
    rng = np.random.default_rng(seed)
    falls = sorted(FALL_ACTIVITIES)
    others = [a for a in ACTIVITIES if a not in FALL_ACTIVITIES]
    sequences = []
    for _ in range(count):
        pool = falls if rng.random() < fall_fraction else others
        sequences.append(generate_sequence(rng, activity=str(rng.choice(pool))))
    return sequences
