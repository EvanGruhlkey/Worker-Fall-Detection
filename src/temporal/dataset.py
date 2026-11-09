"""Cut labeled training windows out of pose sequences.

Follows section 4 of the system guide: slide a 30-frame window over every
person's sequence with a stride of 5, label a window positive when more than
half of a labeled fall interval lies inside it, normalize exactly as in stage
5 (by reusing stage 5), and split by video rather than by window.

Sequences come either from ``temporal.synthetic`` or from pose JSON files
(``yolo_pose.py`` or the YOLO26n + RTMPose pipeline; see ``pose_io``), listed
in a manifest that ``scripts/build_manifest.py`` writes from the label sheet:

    {
      "target_fps": 12,
      "videos": [
        {"name": "clip01", "group": "deck_a", "pose_json": "data/output/clip01.json",
         "falls": [[310, 330]]},                 # source frame indices, inclusive
        {"name": "clip02", "group": "deck_a", "pose_json": "data/output/clip02.json",
         "falls_seconds": [[12.4, 13.1]], "track_ids": [3]}
      ]
    }

Fall times are mapped onto the sampled frames by timestamp, so they stay right
whatever rate the pose file was sampled at. ``track_ids`` limits a video to the
person who falls; when the tracker gives that person a new ID mid-fall, the
new ID inherits the window exactly as it does in live detection.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.sliding_window import TrackWindows

from .detector import handoff_lost_tracks
from .features import extract_features
from .pose_io import load_pose_json


@dataclass
class WindowSample:
    features: np.ndarray  # (window_size, NUM_FEATURES)
    mask: np.ndarray  # (window_size,)
    label: int
    source: str  # video or sequence the window came from
    end_frame: int
    track_id: int
    group: str = ""  # videos of the same scene share a group; splits keep it whole

    @property
    def split_key(self):
        return self.group or self.source


def fall_overlap(end_frame, window_size, fall_intervals):
    """Largest share (0 to 1) of any one fall interval that lies inside the window."""
    start = end_frame - window_size + 1
    best = 0.0
    for first, last in fall_intervals:
        length = last - first + 1
        if length <= 0:
            continue
        overlap = min(last, end_frame) - max(first, start) + 1
        best = max(best, max(overlap, 0) / length)
    return best


def label_for_window(end_frame, window_size, fall_intervals):
    """1 when more than half of any fall interval is inside the window."""
    return int(fall_overlap(end_frame, window_size, fall_intervals) > 0.5)


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
    handoff=True,
    ignore_partial=False,
):
    """Replay per-frame ``people`` lists through stage 5 and cut windows.

    With ``handoff`` (the default, matching live detection) a new track ID that
    appears next to one that was just lost takes over its window; when the lost
    ID was one of ``track_ids``, the new ID is followed from then on.

    With ``ignore_partial`` a window holding some but not more than half of a
    fall is left out, rather than labeled "no fall": it shows the start (or the
    end) of a real fall, and teaching the model that this is normal motion is
    label noise.
    """
    windows = TrackWindows(window_size=window_size, max_gap=max_gap)
    wanted = set(track_ids) if track_ids else None
    samples = []
    for index, people in enumerate(frames):
        if handoff:
            for old_id, new_id in handoff_lost_tracks(windows, people, index):
                if wanted is not None and old_id in wanted:
                    wanted.add(new_id)
        updated = windows.update(index, people)
        if index % stride:
            continue
        for track_id in updated:
            if wanted is not None and track_id not in wanted:
                continue
            window = windows.window(track_id)
            if full_only and not window.is_full:
                continue
            overlap = fall_overlap(index, window_size, fall_intervals)
            if ignore_partial and 0.0 < overlap <= 0.5:
                continue
            samples.append(
                WindowSample(
                    features=extract_features(window, fps),
                    mask=window.mask.copy(),
                    label=int(overlap > 0.5),
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


def samples_from_pose_json(path, falls=(), falls_seconds=(), *, target_fps=12.0, name=None, **options):
    """Windows from one pose JSON file, sampled down to about ``target_fps``.

    ``falls`` are source-frame intervals and ``falls_seconds`` are intervals in
    seconds, both inclusive. Returns ``(samples, fps)`` where ``fps`` is the rate
    the windows actually have (a file sampled at 12 fps stays at 12 even when
    15 is asked for; frames are never invented).
    """
    path = Path(path)
    poses = load_pose_json(path).resample(target_fps)
    seconds = [(a / poses.source_fps, b / poses.source_fps) for a, b in falls]
    seconds += [(float(a), float(b)) for a, b in falls_seconds]
    intervals = [poses.interval_indices(a, b) for a, b in seconds]
    samples = windows_from_frames(
        poses.people, poses.fps, intervals, source=name or path.stem, **options
    )
    return samples, poses.fps


@dataclass
class Clip:
    """One labeled video, already sampled: what cross-validation splits and replays."""

    name: str
    group: str
    fps: float
    frames: list  # per frame, the list of person dicts
    times: list  # seconds from the start of the video, per frame
    fall_intervals: list  # (first, last) frame indices, inclusive
    track_ids: list | None  # the person who falls; None = everyone

    @property
    def has_fall(self):
        return bool(self.fall_intervals)

    @property
    def duration(self):
        return len(self.frames) / self.fps

    @property
    def fall_seconds(self):
        return [(self.times[a], self.times[b]) for a, b in self.fall_intervals]


def load_clips(path):
    """Every video in a manifest as a ``Clip``; returns ``(clips, fps)``.

    All videos must end up at the same frame rate, since one model is trained
    on all of them and live detection samples at the rate stored with it.
    """
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    target_fps = float(manifest.get("target_fps", 12.0))
    clips = []
    for video in manifest["videos"]:
        pose_json = Path(video["pose_json"])
        if not pose_json.is_absolute():
            pose_json = path.parent / pose_json
        name = video.get("name") or pose_json.stem
        poses = load_pose_json(pose_json).resample(target_fps)
        seconds = [(a / poses.source_fps, b / poses.source_fps) for a, b in video.get("falls", ())]
        seconds += [(float(a), float(b)) for a, b in video.get("falls_seconds", ())]
        clips.append(
            Clip(
                name=name,
                group=video.get("group") or name,
                fps=poses.fps,
                frames=poses.people,
                times=poses.times,
                fall_intervals=[poses.interval_indices(a, b) for a, b in seconds],
                track_ids=video.get("track_ids"),
            )
        )
    if not clips:
        raise ValueError(f"{path}: manifest lists no videos")
    fps = min(clip.fps for clip in clips)
    mismatched = {clip.name: clip.fps for clip in clips if abs(clip.fps - fps) > 0.02 * fps}
    if mismatched:
        listing = ", ".join(f"{name} {rate:g}" for name, rate in sorted(mismatched.items()))
        raise ValueError(
            f"Videos end up at different frame rates ({fps:g} fps and {listing}); "
            "re-extract them at one rate or lower target_fps in the manifest"
        )
    return clips, fps


def synthetic_clips(count, fps, seed=0, fall_fraction=0.4):
    """Procedural stick-figure clips (``temporal.synthetic``) at the real data's rate.

    For training only: each is its own group, has one person, and is labeled
    from the generator's known fall interval.
    """
    from .synthetic import ACTIVITIES, FALL_ACTIVITIES, generate_sequence

    rng = np.random.default_rng(seed)
    falls = sorted(FALL_ACTIVITIES)
    others = [a for a in ACTIVITIES if a not in FALL_ACTIVITIES]
    clips = []
    for index in range(count):
        pool = falls if rng.random() < fall_fraction else others
        sequence = generate_sequence(rng, activity=str(rng.choice(pool)), fps=fps)
        name = f"synthetic-{index}-{sequence.activity}"
        clips.append(
            Clip(
                name=name,
                group=name,
                fps=sequence.fps,
                frames=sequence.frames,
                times=[i / sequence.fps for i in range(len(sequence.frames))],
                fall_intervals=[sequence.fall_interval] if sequence.fall_interval else [],
                track_ids=None,
            )
        )
    return clips


def samples_from_clip(clip, **options):
    """Training windows of one clip, tagged with its group."""
    samples = windows_from_frames(
        clip.frames, clip.fps, clip.fall_intervals, source=clip.name, track_ids=clip.track_ids, **options
    )
    for sample in samples:
        sample.group = clip.group
    return samples


def samples_from_manifest(path, **options):
    """Windows from every video in a manifest; returns ``(samples, fps)``."""
    clips, fps = load_clips(path)
    samples = []
    for clip in clips:
        samples += samples_from_clip(clip, **options)
    return samples, fps


def split_by_source(samples, seed=0, val_fraction=0.15, test_fraction=0.15):
    """Assign whole groups (else whole sources) to train / validation / test.

    Never single windows, and never one scene on both sides: near-identical
    clips of one scene would otherwise make the test score look better than
    the model is.
    """
    keys = sorted({sample.split_key for sample in samples})
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    n_test = int(round(len(keys) * test_fraction))
    n_val = int(round(len(keys) * val_fraction))
    test = set(keys[:n_test])
    val = set(keys[n_test:n_test + n_val])
    train = [s for s in samples if s.split_key not in test and s.split_key not in val]
    return (
        train,
        [s for s in samples if s.split_key in val],
        [s for s in samples if s.split_key in test],
    )


def stack(samples):
    """Return arrays X (N, T, F) float32, M (N, T) bool, y (N,) float32."""
    if not samples:
        raise ValueError("No samples to stack")
    features = np.stack([s.features for s in samples]).astype(np.float32)
    masks = np.stack([s.mask for s in samples]).astype(bool)
    labels = np.array([s.label for s in samples], np.float32)
    return features, masks, labels
