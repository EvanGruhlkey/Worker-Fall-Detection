"""Read pose JSON files from either pose pipeline into one shape.

Two writers produce pose files, with the same per-person fields (``track_id``,
``bbox``, ``confidence``, 12 body ``keypoints``) but different frame bookkeeping:

- ``yolo_pose.py`` (and the backend's upload jobs) keep every source frame:
  ``metadata.fps`` is the video's rate and each frame has ``frame_index``.
- The YOLO26n + ByteTrack + RTMPose pipeline samples while extracting:
  ``metadata.fps`` is the sampled rate (e.g. 12), ``metadata.source_fps`` the
  video's rate (e.g. 24), and each frame has ``sample_index``,
  ``source_frame`` and ``timestamp``.

``load_pose_json`` returns a ``PoseFrames`` holding the frames in order with
their time in seconds and their actual rate, so callers never have to know
which writer made the file. Fall labels are given in seconds or source frames,
and are mapped onto frames by time with ``PoseFrames.interval_indices``.
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PoseFrames:
    fps: float  # rate of `people` / `times`
    source_fps: float  # rate of the original video
    times: list  # seconds from the start of the video, one per frame
    people: list  # per frame, the list of person dicts

    def __len__(self):
        return len(self.people)

    def resample(self, target_fps):
        """Keep every n-th frame to get as close to ``target_fps`` as possible.

        Frames are never invented: asking for more than ``fps`` keeps them all.
        """
        if target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")
        stride = max(1, round(self.fps / target_fps))
        return PoseFrames(
            fps=self.fps / stride,
            source_fps=self.source_fps,
            times=self.times[::stride],
            people=self.people[::stride],
        )

    def interval_indices(self, start_seconds, end_seconds):
        """Inclusive frame indices covering a time interval.

        A frame belongs to the interval when its time is within half a frame of
        it, so an interval shorter than one frame still maps to one frame.
        """
        if end_seconds < start_seconds:
            raise ValueError(f"Interval ends before it starts: {start_seconds} > {end_seconds}")
        if not self.times:
            raise ValueError("No frames")
        half = 0.5 / self.fps
        inside = [i for i, t in enumerate(self.times) if start_seconds - half <= t <= end_seconds + half]
        if inside:
            return inside[0], inside[-1]
        middle = (start_seconds + end_seconds) / 2
        nearest = min(range(len(self.times)), key=lambda i: abs(self.times[i] - middle))
        return nearest, nearest


def load_pose_json(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    frames = data.get("frames", [])
    keypoint_format = metadata.get("keypoint_format")
    if keypoint_format not in (None, "COCO-17", "COCO-body-12"):
        raise ValueError(f"Unsupported keypoint format: {keypoint_format!r}")
    expected = {"COCO-17": 17, "COCO-body-12": 12}.get(keypoint_format)
    for frame in frames:
        for person in frame["people"]:
            points = person["keypoints"]
            if len(points) not in (12, 17) or (expected is not None and len(points) != expected):
                raise ValueError("Pose JSON must contain COCO-17 or COCO-body-12 keypoints")
            if len(points) == 17:
                person["keypoints"] = points[5:]
    if not frames:
        raise ValueError(f"{path}: no frames")
    if "source_frame" in frames[0]:
        return _sampled_format(path, metadata, frames)
    return _every_frame_format(path, metadata, frames)


def _sampled_format(path, metadata, frames):
    frames = sorted(frames, key=lambda f: f["source_frame"])
    source_fps = float(metadata.get("source_fps") or 0)
    if source_fps <= 0:
        raise ValueError(f"{path}: sampled pose file without metadata.source_fps")
    steps = {b["source_frame"] - a["source_frame"] for a, b in zip(frames, frames[1:])}
    if len(steps) > 1:
        # Windows count frames, so an uneven step would silently stretch time.
        raise ValueError(f"{path}: frames are not evenly sampled (source frame steps {sorted(steps)})")
    step = steps.pop() if steps else 1
    fps = source_fps / step
    stated = metadata.get("fps")
    if stated and abs(float(stated) - fps) > 0.01 * fps:
        raise ValueError(f"{path}: metadata.fps is {stated} but frames are {fps:g} fps apart")
    return PoseFrames(
        fps=fps,
        source_fps=source_fps,
        times=[f["source_frame"] / source_fps for f in frames],
        people=[f["people"] for f in frames],
    )


def _every_frame_format(path, metadata, frames):
    frames = sorted(frames, key=lambda f: f["frame_index"])
    fps = float(metadata.get("fps") or 30.0)
    indices = [f["frame_index"] for f in frames]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError(f"{path}: frame_index values are not consecutive")
    return PoseFrames(
        fps=fps,
        source_fps=fps,
        times=[i / fps for i in indices],
        people=[f["people"] for f in frames],
    )
