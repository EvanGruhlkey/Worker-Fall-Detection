"""Read pose JSON files from either pose pipeline into one shape.

Two writers produce pose files, with the same per-person fields (``track_id``,
``bbox``, ``confidence``, 17 ``keypoints``) but different frame bookkeeping:

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
        """Bring the frames to ``target_fps``.

        - A rate that is a whole multiple of the target (24 -> 12) keeps every
          n-th frame, so real frames are used unchanged.
        - Any other higher rate (20 -> 12) is interpolated: each person's box
          and keypoints are computed at exact 1/target-second steps from the
          two frames around that time. Picking the nearest frame instead would
          jitter the spacing between frames, and velocities with it.
        - Frames are never invented beyond the source rate: asking for more
          than ``fps`` keeps every frame as it is.
        """
        if target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")
        ratio = self.fps / target_fps
        if ratio <= 1.0 + 1e-6 or abs(ratio - round(ratio)) < 0.02:
            stride = max(1, round(ratio))
            return PoseFrames(
                fps=self.fps / stride,
                source_fps=self.source_fps,
                times=self.times[::stride],
                people=self.people[::stride],
            )
        start, end = self.times[0], self.times[-1]
        count = int((end - start) * target_fps + 1e-6) + 1
        times, people = [], []
        for step in range(count):
            t = start + step / target_fps
            position = (t - start) * self.fps
            before = min(int(position), len(self.people) - 1)
            after = min(before + 1, len(self.people) - 1)
            people.append(_blend(self.people[before], self.people[after], position - before))
            times.append(t)
        return PoseFrames(fps=float(target_fps), source_fps=self.source_fps, times=times, people=people)

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


def _blend(frame_a, frame_b, weight):
    """People at a time ``weight`` (0-1) of the way from frame_a to frame_b.

    A track in both frames is interpolated (keypoint confidence takes the
    lower of the two, so a joint unseen in either stays unseen); a track in
    only one frame is taken from the nearer frame if it has it, else dropped.
    """
    by_id_b = {p.get("track_id"): p for p in frame_b if p.get("track_id") is not None}
    by_id_a = {p.get("track_id"): p for p in frame_a if p.get("track_id") is not None}
    nearer = frame_a if weight < 0.5 else frame_b
    out = []
    for person in nearer:
        track_id = person.get("track_id")
        a, b = by_id_a.get(track_id), by_id_b.get(track_id)
        if track_id is None or a is None or b is None:
            out.append(person)
            continue
        mix = lambda x, y: x + (y - x) * weight  # noqa: E731
        out.append({
            "track_id": track_id,
            "bbox": [mix(x, y) for x, y in zip(a["bbox"], b["bbox"])],
            "confidence": min(a.get("confidence", 1.0), b.get("confidence", 1.0)),
            "keypoints": [
                {"x": mix(ka["x"], kb["x"]), "y": mix(ka["y"], kb["y"]),
                 "confidence": min(ka["confidence"], kb["confidence"])}
                for ka, kb in zip(a["keypoints"], b["keypoints"])
            ],
        })
    return out


def load_pose_json(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = data.get("metadata", {})
    frames = data.get("frames", [])
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
