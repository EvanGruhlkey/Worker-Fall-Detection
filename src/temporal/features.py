"""Classifier input features derived from a sliding-window ``Window``.

The system guide (section 5.1) asks for joint velocities and accelerations as
explicit inputs, not only positions: a GRU can learn frame-to-frame deltas on
its own, but it learns far faster when handed them. Everything here is computed
from the ``Window`` that stage 5 produces, so training data and live inference
go through exactly the same code (no train/serve skew).

Per frame the feature vector is, in order:

| block          | size | meaning                                                   |
|----------------|------|-----------------------------------------------------------|
| positions      | 36   | 12 joints x (x, y, confidence), normalized by stage 5     |
| velocities     | 24   | 12 joints x (dx, dy) per second, body heights / s         |
| accelerations  | 24   | 12 joints x (ddx, ddy) per second^2                       |
| box            | 2    | box height / reference height, box width / height         |
| box velocity   | 2    | per-second change of the two box features                 |
| signals        | 3    | the interpretable motion signals listed in SIGNAL_NAMES   |

Velocities are in body heights per second so that the same value means the
same motion at 10, 15 or 25 frames per second; the frame rate is passed in.
Padded frames (mask False) are all zero, and no velocity is computed across a
padding boundary.
"""

import numpy as np

from sliding_window import NUM_KEYPOINTS

# Bump when the layout or meaning of the feature vector changes, so an old
# checkpoint is refused rather than silently fed different numbers.
FEATURE_VERSION = 2

# COCO body points after removing facial keypoints 0-4.
LEFT_SHOULDER, RIGHT_SHOULDER = 0, 1
LEFT_HIP, RIGHT_HIP = 6, 7

# YOLO still reports a position for joints it cannot see, with low confidence
# and a lot of jitter. Below this the position is treated as missing and the
# last good position is carried forward (the guide's rule), which keeps the
# velocity channels free of jumps that are not motion.
MIN_KEYPOINT_CONFIDENCE = 0.3

SIGNAL_NAMES = (
    "hip_height",     # hip centre above the box bottom, in body heights
    "torso_angle",    # 0 = upright, 1 = horizontal (degrees / 90)
    "hip_velocity",   # upward hip speed in body heights / s (a fall is negative)
)
NUM_SIGNALS = len(SIGNAL_NAMES)
SIGNAL_INDEX = {name: index for index, name in enumerate(SIGNAL_NAMES)}

_POSITIONS = NUM_KEYPOINTS * 3
_VELOCITIES = NUM_KEYPOINTS * 2
_BOX = 2
NUM_FEATURES = _POSITIONS + 2 * _VELOCITIES + 2 * _BOX + NUM_SIGNALS  # 91
SIGNAL_SLICE = slice(NUM_FEATURES - NUM_SIGNALS, NUM_FEATURES)


def feature_names():
    """Human-readable name of every column of the feature vector."""
    names = []
    for joint in range(NUM_KEYPOINTS):
        names += [f"kp{joint}_x", f"kp{joint}_y", f"kp{joint}_conf"]
    for prefix in ("vel", "acc"):
        for joint in range(NUM_KEYPOINTS):
            names += [f"{prefix}{joint}_x", f"{prefix}{joint}_y"]
    names += ["box_height", "box_aspect", "box_height_vel", "box_aspect_vel"]
    names += list(SIGNAL_NAMES)
    assert len(names) == NUM_FEATURES
    return names


def extract_features(window, fps):
    """Return the (window_size, NUM_FEATURES) float32 feature matrix."""
    if window.boxes is None:
        raise ValueError("Window has no boxes; build windows with TrackWindows")
    if fps <= 0:
        raise ValueError("fps must be greater than zero")

    mask = np.asarray(window.mask, dtype=bool)
    frames = len(mask)
    confidence = window.keypoints[..., 2].astype(np.float32)
    xy = fill_missing_joints(window.keypoints[..., :2], confidence, mask, window.boxes)

    # Both frames of a pair must be real; a delta across padding is not motion.
    pair = mask[1:] & mask[:-1]
    velocity = np.zeros((frames, NUM_KEYPOINTS, 2), np.float32)
    velocity[1:][pair] = (xy[1:] - xy[:-1])[pair] * fps
    both = pair[1:] & pair[:-1]
    acceleration = np.zeros_like(velocity)
    acceleration[2:][both] = (velocity[2:] - velocity[1:-1])[both] * fps

    box_velocity = np.zeros((frames, _BOX), np.float32)
    box_velocity[1:][pair] = (window.box_features[1:] - window.box_features[:-1])[pair] * fps

    signals = compute_signals(xy, confidence, window.boxes, mask, fps)
    positions = np.concatenate([xy, confidence[..., None]], axis=-1).reshape(frames, -1)
    features = np.concatenate(
        [
            positions,
            velocity.reshape(frames, -1),
            acceleration.reshape(frames, -1),
            window.box_features,
            box_velocity,
            signals,
        ],
        axis=1,
    ).astype(np.float32)
    features[~mask] = 0.0
    return features


def window_signals(window, fps):
    """The SIGNAL_NAMES values for the newest frame of a window (for plots)."""
    mask = np.asarray(window.mask, dtype=bool)
    confidence = window.keypoints[..., 2].astype(np.float32)
    xy = fill_missing_joints(window.keypoints[..., :2], confidence, mask, window.boxes)
    return compute_signals(xy, confidence, window.boxes, mask, fps)[-1]


def fill_missing_joints(xy, confidence, mask, boxes):
    """Replace low-confidence joint positions with the last good one.

    A joint that has not been seen yet takes its first later sighting, and a
    joint never seen in the window sits at that frame's box centre. Only real
    frames are touched; padded frames stay zero.
    """
    filled = np.array(xy, dtype=np.float32, copy=True)
    visible = confidence >= MIN_KEYPOINT_CONFIDENCE
    real = np.flatnonzero(mask)
    if len(real) == 0:
        return filled

    held = np.zeros((NUM_KEYPOINTS, 2), np.float32)
    known = np.zeros(NUM_KEYPOINTS, dtype=bool)
    for frame in real:
        seen = visible[frame]
        held[seen] = xy[frame][seen]
        known |= seen
        center = np.array(
            [(boxes[frame, 0] + boxes[frame, 2]) / 2, (boxes[frame, 1] + boxes[frame, 3]) / 2],
            np.float32,
        )
        filled[frame] = np.where(known[:, None], held, center)

    # Frames before a joint's first sighting hold that first position instead
    # of the box centre, so the joint does not "jump" when it appears.
    for joint in range(NUM_KEYPOINTS):
        sightings = real[visible[real, joint]]
        if len(sightings):
            first = sightings[0]
            filled[real[real < first], joint] = xy[first, joint]
    return filled


def compute_signals(xy, confidence, boxes, mask, fps):
    """Return (frames, NUM_SIGNALS) interpretable motion signals.

    Heights are measured from the bottom of the person's own box, so they do
    not depend on where the window's origin is, and velocities are upward
    positive so a fall shows as a negative spike, as in the guide's figure.
    """
    mask = np.asarray(mask, dtype=bool)
    visible = confidence >= MIN_KEYPOINT_CONFIDENCE
    hip = _visible_midpoint(xy, visible, (LEFT_HIP, RIGHT_HIP))
    shoulder = _visible_midpoint(xy, visible, (LEFT_SHOULDER, RIGHT_SHOULDER))

    bottom = boxes[:, 3]
    hip_height = bottom - hip[:, 1]

    # Image y grows downward, so dy > 0 means the shoulders are above the hips.
    dx = shoulder[:, 0] - hip[:, 0]
    dy = hip[:, 1] - shoulder[:, 1]
    torso_angle = np.degrees(np.arctan2(np.abs(dx), dy)) / 90.0

    pair = mask[1:] & mask[:-1]
    hip_velocity = np.zeros(len(mask), np.float32)
    hip_velocity[1:][pair] = -(hip[1:, 1] - hip[:-1, 1])[pair] * fps

    signals = np.stack(
        [hip_height, torso_angle, hip_velocity], axis=1
    ).astype(np.float32)
    signals[~mask] = 0.0
    return signals


def _visible_midpoint(xy, visible, joints):
    """Mean of the visible joints among `joints`, or of all of them if none is."""
    points = xy[:, list(joints)]  # (frames, n, 2)
    weights = visible[:, list(joints)].astype(np.float32)[..., None]
    total = weights.sum(axis=1)
    weighted = (points * weights).sum(axis=1) / np.maximum(total, 1.0)
    plain = points.mean(axis=1)
    return np.where(total > 0, weighted, plain)
