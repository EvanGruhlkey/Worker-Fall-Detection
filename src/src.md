# Source guide

Start with `fall_pipeline.py`. It reads video or saved pose data and connects pose tracking to the sliding window and temporal fall classifier.

```text
src/
├── detectors/
│   ├── yolo_detector.py
│   └── yolo_pose.py
├── pose_estimators/
│   ├── rtmpose_estimator.py
│   ├── rtmpose_pipeline.py
│   └── pose_runtime.py
├── trackers/
│   └── bytetrack_tracker.py
├── utils/
│   └── sliding_window.py
├── fall_pipeline.py
└── src.md
```

The existing `backend/` and `temporal/` folders contain the API and fall classifier.

## How a frame moves through the pipeline

`detectors/yolo_pose.py` loads `models/pose_pipeline.json`. The config names the two ONNX models. `pose_estimators/rtmpose_pipeline.py` calls these steps in order:

1. `detectors/yolo_detector.py` finds person bounding boxes.
2. `trackers/bytetrack_tracker.py` assigns tracking IDs across frames.
3. `pose_estimators/rtmpose_estimator.py` finds 12 body keypoints for each tracked person.

The pipeline returns an annotated frame and people with `track_id`, `bbox`, `confidence`, and `keypoints`.

`fall_pipeline.py` passes those people to `utils/sliding_window.py`, then scores each full window with the temporal classifier and applies the alert threshold.

`pose_estimators/pose_runtime.py` chooses NVIDIA CUDA, then DirectML, then CPU for ONNX models, based on installed runtime support. The detector and RTMPose estimator share it.

## Run it

From the repository root:

```sh
uv run src/fall_pipeline.py --input data/input/test.mp4
uv run src/detectors/yolo_pose.py --model models/pose_pipeline.json
uv run src/detectors/yolo_pose.py --model models/pose_pipeline.json --input data/input/test.mp4
```

The backend imports the same pose functions. Python imports use the folders, for example `from utils.sliding_window import TrackWindows`.

## Device selection

`choose_provider()` in `pose_estimators/pose_runtime.py` selects the first available option:

1. CUDA for NVIDIA GPUs.
2. DirectML for compatible Windows GPUs, including AMD.
3. CPU everywhere else.

This is a preference order, not a speed benchmark. Both ONNX models use the same selection rule and allow CPU fallback. Without a `device_id` in the JSON config, DirectML uses the high-performance GPU: on laptops with built-in graphics plus a separate GPU (e.g. Intel Iris Xe + NVIDIA) that is the separate one, about twice as fast as GPU 0, which is usually the built-in one. CUDA uses GPU 0. Add `"device_id": N` to the config to pick a GPU by number. Use `"backend": "rtmpose"` in new configs; the old `"directml_rtmpose"` value also works with automatic selection.

The default `uv sync` installs DirectML on Windows and CPU runtime elsewhere. On a machine with NVIDIA CUDA and cuDNN installed, use the CUDA group when running commands:

```sh
uv run --no-group runtime --group cuda run.py
uv run --no-group runtime --group cuda src/fall_pipeline.py --input data/input/test.mp4
```

The groups prevent installing multiple ONNX Runtime packages into the same environment. Device selection uses the installed runtime's available providers.

The pose pipeline uses ONNX and emits only 12 body keypoints. Head points and head-based temporal signals are excluded. PyTorch checkpoints remain available for training and for the separate temporal fall classifier.
