"""Cut labeled training windows out of pose sequences.

Follows section 4 of the system guide: slide a 30-frame window over every
person's sequence with a stride of 5, label a window positive when more than
half of a labeled fall interval lies inside it, normalize exactly as in stage
5 (by reusing stage 5), and split by video rather than by window.

Sequences come either from ``temporal.synthetic`` or from the pose JSON files
that ``yolo_pose.py`` writes, listed in a manifest:

    {
      "target_fps": 15,
      "videos": [
        {"name": "clip01", "pose_json": "data/output/clip01.json",
         "falls": [[310, 330]]},                 # source frame indices, inclusive
        {"name": "clip02", "pose_json": "data/output/clip02.json",
         "falls_seconds": [[12.4, 13.1]], "track_ids": [3]}
      ]
    }
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sliding_window import TrackWindows

from .features import extract_features
from .pose_io import load_pose_json
from .sampler import FrameSampler


@dataclass
class WindowSample:
    features: np.ndarray  # (window_size, NUM_FEATURES)
    mask: np.ndarray  # (window_size,)
    label: int
    source: str  # video or sequence the window came from, for splitting
    end_frame: int
    track_id: int


def label_for_window(end_frame, window_size, fall_intervals):
    """1 when more than half of any fall interval is inside the window."""
    start = end_frame - window_size + 1
    for first, last in fall_intervals:
        length = last - first + 1
        overlap = min(last, end_frame) - max(first, start) + 1
        if length > 0 and overlap / length > 0.5:
            return 1
    return 0


def windows_from_frames(
    frames,
    fps,
    fall_intervals=(),
    *,
    source="",
    stride=5,
    window_size=30,
    max_gap=5,
    track_ids=None,
    full_only=True,
):
    """Replay per-frame ``people`` lists through stage 5 and cut windows."""
    windows = TrackWindows(window_size=window_size, max_gap=max_gap)
    samples = []
    for index, people in enumerate(frames):
        updated = windows.update(index, people)
        if index % stride:
            continue
        for track_id in updated:
            if track_ids and track_id not in track_ids:
                continue
            window = windows.window(track_id)
            if full_only and not window.is_full:
                continue
            samples.append(
                WindowSample(
                    features=extract_features(window, fps),
                    mask=window.mask.copy(),
                    label=label_for_window(index, window_size, fall_intervals),
                    source=source,
                    end_frame=index,
                    track_id=track_id,
                )
            )
    return samples


def samples_from_sequences(sequences, **options):
    samples = []
    for index, sequence in enumerate(sequences):
        intervals = [sequence.fall_interval] if sequence.fall_interval else []
        samples += windows_from_frames(
            sequence.frames,
            sequence.fps,
            intervals,
            source=f"synthetic-{index}-{sequence.activity}",
            **options,
        )
    return samples


def samples_from_pose_json(path, falls=(), falls_seconds=(), *, target_fps=15.0, name=None, **options):
    """Windows from a ``yolo_pose.py`` JSON file, resampled to ``target_fps``."""
    path = Path(path)
    data = load_pose_json(path)
    source_fps = float(data["metadata"].get("fps") or 30.0)
    sampler = FrameSampler(source_fps, target_fps)

    intervals = [(int(a), int(b)) for a, b in falls]
    intervals += [(int(round(a * source_fps)), int(round(b * source_fps))) for a, b in falls_seconds]
    # Fall intervals are given in source frames; windows count kept frames.
    kept_intervals = [(a // sampler.stride, b // sampler.stride) for a, b in intervals]

    frames = [
        frame["people"]
        for frame in sorted(data["frames"], key=lambda f: f["frame_index"])
        if sampler.keep(frame["frame_index"])
    ]
    return windows_from_frames(
        frames, sampler.fps, kept_intervals, source=name or path.stem, **options
    )


def samples_from_manifest(path, **options):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    target_fps = float(manifest.get("target_fps", 15.0))
    samples = []
    for video in manifest["videos"]:
        pose_json = Path(video["pose_json"])
        if not pose_json.is_absolute():
            pose_json = path.parent / pose_json
        samples += samples_from_pose_json(
            pose_json,
            falls=video.get("falls", ()),
            falls_seconds=video.get("falls_seconds", ()),
            target_fps=target_fps,
            name=video.get("name"),
            track_ids=video.get("track_ids"),
            **options,
        )
    return samples, target_fps


def split_by_source(samples, seed=0, val_fraction=0.15, test_fraction=0.15):
    """Assign whole sources to train / validation / test, never single windows."""
    sources = sorted({sample.source for sample in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(sources)
    n_test = int(round(len(sources) * test_fraction))
    n_val = int(round(len(sources) * val_fraction))
    test = set(sources[:n_test])
    val = set(sources[n_test:n_test + n_val])
    train = [s for s in samples if s.source not in test and s.source not in val]
    return (
        train,
        [s for s in samples if s.source in val],
        [s for s in samples if s.source in test],
    )


def stack(samples):
    """Return arrays X (N, T, F) float32, M (N, T) bool, y (N,) float32."""
    if not samples:
        raise ValueError("No samples to stack")
    features = np.stack([s.features for s in samples]).astype(np.float32)
    masks = np.stack([s.mask for s in samples]).astype(bool)
    labels = np.array([s.label for s in samples], np.float32)
    return features, masks, labels
