# 10-18-2026 - Sliding window buffer (pipeline stage 5)

## Goal

A single frame cannot show a fall: a person low to the ground could be
falling, crouching, or working under equipment. A fall is a motion over time,
so the temporal classifier (stage 6, not built yet) needs a short history of
each person's body positions, not one frame.

The sliding window keeps the most recent 30 frames of keypoints for every
tracked person. Each new frame is added and the oldest frame drops out, and
the whole window is handed to the classifier.

```
YOLO pose (stage 3) -> ByteTrack (stage 4) -> TrackWindows (stage 5) -> classifier (stage 6)
  bbox + 12 keypoints     + track_id            (30, 12, 3) per person     fall probability
```

## Implementation

**`src/utils/sliding_window.py`**, plain NumPy with no model code.

- **One queue per person.** A `dict` maps each ByteTrack `track_id` to a
  `collections.deque(maxlen=30)`. A full deque discards its oldest item
  automatically on append, so each update is O(1) per person. People never
  share a window, so two workers crossing paths are never mixed together.
- **Same input format as `yolo_pose.py`.** `update()` takes the `people` list
  that `process_frame` returns, which is also the format saved to JSON. Live
  inference and offline training data go through the same code, so there is
  no train/serve skew.
- **Track gaps.** ByteTrack can lose a person for a few frames and give them
  back the same ID. Gaps of up to `max_gap` (5) frames are filled by repeating
  the last pose with confidence 0. A longer gap restarts the window, because
  motion across it is unreliable.
- **Stale tracks.** Tracks not seen for `stale_after` (30) frames are dropped
  so memory does not grow as people come and go. This matches ByteTrack's
  default `track_buffer: 30`, after which it will not reuse the ID.
- **Short tracks.** Windows with fewer than 30 frames are left-padded with
  zeros and come with a `mask` marking which frames are real, so the
  classifier can ignore the padding.
- **Normalization at read time.** Raw pixels are stored and normalized only
  when a window is read, so normalization can change without touching the
  buffer. Every frame in a window uses one shared reference:
  - origin: center of the newest frame's bounding box (position in the image
    does not matter);
  - scale: tallest bounding box in the window, roughly standing height
    (distance from the camera does not matter).

  Normalizing each frame by its own box would erase the fall itself: the hips
  stay centered in their box while the box drops.
- **Box features.** Each frame also carries bbox height relative to the
  reference height and bbox width / height. A body that suddenly gets shorter
  and wider is itself a fall signal.
- **Validation.** A malformed frame (wrong keypoint count, bad bbox, same
  track ID twice, frames out of order) raises `ValueError` before anything is
  stored, so the buffer is never left half updated.

### Output per person

| Field          | Shape            | Meaning                                         |
|----------------|------------------|-------------------------------------------------|
| `keypoints`    | (30, 12, 3)      | normalized x, y and raw keypoint confidence     |
| `flatten()`    | (30, 36)         | classifier input                                |
| `box_features` | (30, 2)          | relative bbox height, bbox width / height       |
| `mask`         | (30,)            | True for real frames, False for padding         |
| `end_frame`    | int              | frame index of the newest frame                 |

### Usage

```python
from utils.sliding_window import TrackWindows

windows = TrackWindows()  # window_size=30, max_gap=5, stale_after=30
for frame_index, frame in enumerate(frames):
    annotated_frame, people = process_frame(model, frame)
    for track_id in windows.update(frame_index, people):
        window = windows.window(track_id)
        if window.is_full:
            x = window.flatten()  # (30, 36)
```

Call `update()` for every processed frame, even one with no people, so missing
tracks are noticed. `frame_index` must count **processed** frames: if a
sampler keeps every 2nd video frame, pass 0, 1, 2, ... rather than
0, 2, 4, ..., or every other step looks like a gap.

## Verification

- **`tests/test_sliding_window.py`**, 25 tests, run with `uv run pytest tests -v`.
  - 24 unit tests on hand-built fake people: frames kept newest-last in
    order, padding, people kept apart, gaps at and past `max_gap`, stale
    tracks, rejected bad input, and normalization (fall preserved, position
    and distance removed, confidence untouched, no NaN on zero-size boxes).
  - 1 integration test replaying real YOLO + ByteTrack output
    (`data/output/001.json`, 1,800 frames) through the window.
- **Mutation check.** Seven bugs were planted in a copy of the module one at
  a time (queue never drops frames, per-frame normalization, gap fill keeping
  confidence, off-by-one `max_gap` and stale limits, no duplicate-ID check,
  padding on the wrong side). The suite failed on every one.
- **Real model run.** `yolo11n-pose.pt` + ByteTrack was run on
  `data/output/001.gif` and fed into the window. The newest frame of every
  window matched YOLO's output exactly, and no window mixed two people (the
  largest hip movement between frames was 0.14 body heights). The window
  costs about 0.27 ms per frame; YOLO costs about 370 ms per frame on a CPU.

## Known limitations

- **Tracking breaks windows on low-frame-rate footage.** On the 10 fps GIF no
  person stayed tracked for 30 frames, because ByteTrack kept assigning new
  IDs. The window correctly restarts on a new ID, so this is an upstream
  tracking issue. The 30 fps replay produced 2,791 full windows. Check
  tracking on real site footage before training.
- **Window length vs frame rate.** The guide assumes 30 frames at 15 fps
  (2 seconds). The camera and video currently run at 30 fps, so a window
  covers 1 second until a frame sampler (stage 2) exists. Training and live
  use must make the same choice.
- **Velocities** (frame-to-frame deltas), which the guide recommends, are not
  computed here; they belong in the classifier's feature step.
- **NumPy** is used directly but is only installed through `ultralytics`.

## How to use going forward

```bash
uv sync                    # installs pytest too (the dev group is installed by default)
uv run pytest tests -v     # run the tests
```
