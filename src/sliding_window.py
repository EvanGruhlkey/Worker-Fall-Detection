"""Stage 5: per-person sliding windows of keypoints for the temporal classifier.

Takes the `people` list produced by `yolo_pose.process_frame` (the same format
that is saved to JSON), so live inference and offline training share this code.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

NUM_KEYPOINTS = 17


@dataclass
class Window:
    """One classifier input for one tracked person.

    keypoints:    (window_size, 17, 3) normalized x, y and raw confidence
    box_features: (window_size, 2) bbox height relative to the window's
                  reference height, and bbox width / height
    mask:         (window_size,) True for real frames, False for left padding
    """

    track_id: int
    end_frame: int
    keypoints: np.ndarray
    box_features: np.ndarray
    mask: np.ndarray

    @property
    def is_full(self):
        return bool(self.mask.all())

    def flatten(self):
        """Return keypoints as (window_size, 51), the classifier's input."""
        return self.keypoints.reshape(len(self.keypoints), -1)


class _Track:
    def __init__(self, window_size):
        self.keypoints = deque(maxlen=window_size)
        self.boxes = deque(maxlen=window_size)
        self.last_frame = None

    def clear(self):
        self.keypoints.clear()
        self.boxes.clear()


class TrackWindows:
    """Keep the most recent `window_size` frames of keypoints per track ID.

    A deque with maxlen drops the oldest frame automatically on append, so each
    update is O(1) per person. Raw pixel values are stored and normalized only
    when a window is read, so normalization can change without touching this.
    """

    def __init__(self, window_size=30, max_gap=5, stale_after=30):
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if max_gap < 0:
            raise ValueError("max_gap must be zero or greater")
        # Otherwise a track would be forgotten before its gap could be filled.
        if stale_after < max_gap:
            raise ValueError("stale_after must be at least max_gap")
        self.window_size = window_size
        # Missing frames up to max_gap are filled by carrying the last pose
        # forward with confidence 0; a longer gap restarts the window.
        self.max_gap = max_gap
        # Forget tracks unseen this many frames. Matches ByteTrack's default
        # track_buffer, after which the tracker will not reuse the ID anyway.
        self.stale_after = stale_after
        self._tracks = {}

    def __len__(self):
        return len(self._tracks)

    def __contains__(self, track_id):
        return track_id in self._tracks

    def update(self, frame_index, people):
        """Add one frame of tracked people. Return the track IDs updated.

        Call this for every processed frame, even one with no people, so
        missing tracks are noticed. `frame_index` must count processed
        frames: if the sampler keeps every 2nd video frame, pass 0, 1, 2, ...
        rather than 0, 2, 4, ..., or every other step looks like a gap.
        """
        # Validate the whole frame first so bad input never leaves a frame
        # half added.
        parsed = []
        for person in people:
            track_id = person["track_id"]
            # The tracker has not confirmed this detection yet.
            if track_id is None:
                continue
            if any(track_id == seen for seen, _, _ in parsed):
                raise ValueError(f"Track ID {track_id} appears twice in one frame")

            keypoints = np.array(
                [[p["x"], p["y"], p["confidence"]] for p in person["keypoints"]],
                dtype=np.float32,
            )
            if keypoints.shape != (NUM_KEYPOINTS, 3):
                raise ValueError("Each person must have exactly 17 keypoints")
            box = np.asarray(person["bbox"], dtype=np.float32)
            if box.shape != (4,):
                raise ValueError("Each bbox must be [x1, y1, x2, y2]")

            track = self._tracks.get(track_id)
            if track is not None and frame_index <= track.last_frame:
                raise ValueError("Frames must be added in increasing order")
            parsed.append((track_id, keypoints, box))

        updated = []
        for track_id, keypoints, box in parsed:
            track = self._tracks.get(track_id)
            if track is None:
                track = self._tracks[track_id] = _Track(self.window_size)
            else:
                self._fill_gap(track, frame_index - track.last_frame - 1)

            track.keypoints.append(keypoints)
            track.boxes.append(box)
            track.last_frame = frame_index
            updated.append(track_id)

        self._drop_stale(frame_index)
        return updated

    def window(self, track_id):
        """Return the current normalized Window for a track, or None."""
        track = self._tracks.get(track_id)
        if track is None or not track.keypoints:
            return None

        keypoints, box_features = normalize_window(
            np.stack(track.keypoints), np.stack(track.boxes)
        )

        # Left-pad short tracks so every window has the same shape.
        padding = self.window_size - len(keypoints)
        mask = np.ones(self.window_size, dtype=bool)
        if padding:
            mask[:padding] = False
            keypoints = np.concatenate(
                [np.zeros((padding, NUM_KEYPOINTS, 3), np.float32), keypoints]
            )
            box_features = np.concatenate(
                [np.zeros((padding, 2), np.float32), box_features]
            )

        return Window(track_id, track.last_frame, keypoints, box_features, mask)

    def _fill_gap(self, track, missing):
        if missing <= 0:
            return
        if missing > self.max_gap:
            track.clear()
            return

        held = track.keypoints[-1].copy()
        held[:, 2] = 0.0
        for _ in range(missing):
            track.keypoints.append(held)
            track.boxes.append(track.boxes[-1])

    def _drop_stale(self, frame_index):
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if frame_index - track.last_frame > self.stale_after
        ]
        for track_id in stale:
            del self._tracks[track_id]


def normalize_window(keypoints, boxes):
    """Normalize raw (T, 17, 3) keypoints and (T, 4) xyxy boxes.

    Every frame uses the same reference point and scale. Normalizing each frame
    by its own box would erase the drop itself: the hips stay centered in their
    box while the box falls. Instead:
      - origin: center of the most recent frame's box, so position in the
        image does not matter
      - scale: tallest box in the window, roughly the person's standing height,
        so distance from the camera does not matter
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    scale = max(float(heights.max()), 1.0)
    center_x = (boxes[-1, 0] + boxes[-1, 2]) / 2
    center_y = (boxes[-1, 1] + boxes[-1, 3]) / 2

    normalized = keypoints.copy()
    normalized[..., 0] = (keypoints[..., 0] - center_x) / scale
    normalized[..., 1] = (keypoints[..., 1] - center_y) / scale

    # A body that suddenly gets shorter and wider is itself a fall signal.
    box_features = np.stack(
        [heights / scale, widths / np.maximum(heights, 1.0)], axis=1
    ).astype(np.float32)
    return normalized, box_features
