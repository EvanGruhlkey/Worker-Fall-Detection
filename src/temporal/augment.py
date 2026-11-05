"""Pose-space augmentation of whole training clips.

Applied to a clip's tracked people before windows are cut, so feature
extraction, track handoff and labeling treat an augmented clip exactly like a
real one. Only training clips are augmented; validation and test never are.

Windows are normalized for position and size (stage 5), so shifting or
scaling a person would change nothing the model sees. What does change it:

- mirror:        left and right swap (camera on the other side)
- rotation:      a small camera tilt, a few degrees
- stretch:       horizontal squeeze or widen (another viewing angle)
- jitter:        keypoint noise, as a less steady pose model produces
- joint dropout: joints briefly unseen (equipment, other workers in front)
- time stretch:  the same motion slightly faster or slower

Each clip draws one mirror / rotation / stretch / speed for its whole length
(it is one camera), and fresh jitter and dropout per frame.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

# COCO-17 left/right keypoint pairs (eyes, ears, shoulders, elbows, wrists,
# hips, knees, ankles).
_PAIRS = ((1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
MIRROR_ORDER = list(range(17))
for _left, _right in _PAIRS:
    MIRROR_ORDER[_left], MIRROR_ORDER[_right] = _right, _left


@dataclass(frozen=True)
class AugmentConfig:
    mirror: float = 0.5  # chance a clip is mirrored
    rotation_degrees: float = 8.0  # uniform in +/- this
    stretch: tuple = (0.85, 1.15)  # horizontal scale factor
    jitter: float = 0.01  # keypoint noise std, in body (box) heights
    joint_dropout: float = 0.03  # chance a joint is unseen in a frame
    speed: tuple = (0.85, 1.15)  # playback speed factor


def augment_clip(clip, rng, config=AugmentConfig()):
    """Return a new, randomly augmented copy of ``clip`` (a ``dataset.Clip``)."""
    mirror = rng.random() < config.mirror
    angle = math.radians(rng.uniform(-config.rotation_degrees, config.rotation_degrees))
    stretch = rng.uniform(*config.stretch)
    speed = rng.uniform(*config.speed)

    cos, sin = math.cos(angle), math.sin(angle)
    sx = -stretch if mirror else stretch

    def point(x, y):
        x = x * sx
        return x * cos - y * sin, x * sin + y * cos

    def person(original):
        x1, y1, x2, y2 = original["bbox"]
        height = max(y2 - y1, 1.0)
        corners = [point(x, y) for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]
        xs, ys = [c[0] for c in corners], [c[1] for c in corners]
        source = [original["keypoints"][j] for j in (MIRROR_ORDER if mirror else range(17))]
        noise = rng.normal(0.0, config.jitter * height, size=(17, 2))
        unseen = rng.random(17) < config.joint_dropout
        keypoints = []
        for j, kp in enumerate(source):
            x, y = point(kp["x"], kp["y"])
            confidence = 0.0 if unseen[j] else kp["confidence"]
            keypoints.append({"x": x + noise[j, 0], "y": y + noise[j, 1], "confidence": confidence})
        return {
            "track_id": original["track_id"],
            "bbox": [min(xs), min(ys), max(xs), max(ys)],
            "confidence": original.get("confidence", 1.0),
            "keypoints": keypoints,
        }

    # Playback speed: new frame i shows source frame round(i * speed).
    length = max(1, int(round(len(clip.frames) / speed)))
    picks = [min(len(clip.frames) - 1, int(round(i * speed))) for i in range(length)]
    frames = [[person(p) for p in clip.frames[source]] for source in picks]
    start = clip.times[0] if clip.times else 0.0
    times = [start + i / clip.fps for i in range(length)]
    intervals = []
    for first, last in clip.fall_intervals:
        a = min(length - 1, int(round(first / speed)))
        b = min(length - 1, max(a, int(round(last / speed))))
        intervals.append((a, b))
    return replace(clip, frames=frames, times=times, fall_intervals=intervals)


def augmented_copies(clips, copies, seed, config=AugmentConfig()):
    """``copies`` augmented versions of every clip (originals not included)."""
    rng = np.random.default_rng(seed)
    out = []
    for clip in clips:
        for index in range(copies):
            out.append(replace(augment_clip(clip, rng, config), name=f"{clip.name}~aug{index}"))
    return out
