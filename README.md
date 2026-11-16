# Oxy Worker Fall Detection

#### Computer vision safety software to detect worker falls



![Full pipeline output](docs/images/pipeline/00-full-pipeline.gif)

## The pipeline

### 1. Video input

OpenCV reads the file or camera one frame at a time.

![Stage 1: video input](docs/images/pipeline/01-video-input.gif)

### 2. Frame sampling

The camera runs at 24 fps. The sampler keeps every 2nd frame (12 fps) and skips the rest. Shown at half speed. 

![Stage 2: frame sampling](docs/images/pipeline/02-frame-sampling.gif)

### 3. YOLO detection

YOLO draws a box and a confidence score around every person in each frame. 

![Stage 3: YOLO detection](docs/images/pipeline/03-yolo-detection.gif)

### 4. ByteTrack

ByteTrack matches boxes across frames so each person keeps an ID. The line behind each box is the person's path. 

![Stage 4: ByteTrack](docs/images/pipeline/04-bytetrack.gif)

### 5. RTMPose

RTMPose finds 12 joints on each tracked person: shoulders, elbows, wrists, hips, knees, ankles. 

![Stage 5: RTMPose](docs/images/pipeline/05-rtmpose.gif)

### 6. Temporal window

For each person, the last 30 frames of joints (2.5 s) are kept, centered and scaled to body height. The panel shows the window for the highlighted person: faint is oldest, bright is newest. 

![Stage 6: temporal window](docs/images/pipeline/06-temporal-window.gif)

### 7. Fall decision

A GRU classifier trained on the labeled rig clips scores each person's window on every frame. A threshold gate raises the alert once the probability stays above 0.5 for 5 frames in a row. The graph shows the highlighted person's probability. 

![Stage 7: fall decision](docs/images/pipeline/07-fall-decision.gif)


### Currently Working

- **Pose detection and tracking** on a live camera or a prerecorded video. The live view shows skeletons, tracking IDs, FPS, and the number of detected people.
- **Backend server** (`src/backend/`). Runs the camera through the models and a placeholder fall detector. Each detected fall is saved to the team's shared PostgreSQL database (Supabase) with its time, camera, location, confidence, a snapshot and a short clip; the snapshot and clip go to shared storage. Every laptop running the project sees every fall. See [docs/database.md](docs/database.md).
- **Temporal classifier** (`src/temporal/`). Stages 2, 6 and 7 of the design: frame sampler, GRU/TCN classifier over each person's 30-frame window of joint positions, velocities and accelerations, and the threshold gate. A bootstrap model trained on synthetic skeletons is in `models/fall_gru.pt`; `src/fall_pipeline.py` runs it over a video and plots every person's hip/head height, torso angle, vertical velocity and fall probability. See [docs/features/10-23-2026-temporal-classifier.md](docs/features/10-23-2026-temporal-classifier.md).
- **Dashboard** (`ui/`). Live annotated camera feed, stat cards (active incidents, cameras online, uptime, camera health), and an event log that updates every few seconds. Click an event to see its clip and confirm it or mark it a false alarm. Upload a recorded video to check it for falls; a test video is in `data/samples/`.

**Full guide to running and testing everything: [docs/running-and-testing.md](docs/running-and-testing.md)**

### Setup

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies. It pins Python 3.12 because PyTorch's last builds for Intel Macs only support up to 3.12, and it locks every dependency in `uv.lock` so everyone gets the same working versions.

Install uv (once):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then from the project root:

```bash
uv sync
```

On Windows this includes DirectML for AMD and other compatible GPUs. On NVIDIA
machines with CUDA/cuDNN installed, run with `uv run --no-group runtime --group cuda`
in place of `uv run`. Other systems use CPU. The YOLO detector and RTMPose models
use ONNX; the optional YOLO-Pose model uses Ultralytics `.pt` weights. See
[src/src.md](src/src.md) for the pipeline and device selection.

This downloads Python 3.12 if needed, creates `.venv`, and installs the locked dependencies. You do not need to activate the virtual env; prefix commands with `uv run` instead.

To add a dependency, use `uv add <package>` (this updates `pyproject.toml` and `uv.lock`). Commit both files.

Events are stored in a shared PostgreSQL database and storage bucket in the cloud (Supabase), so there is nothing to install for them. Put the team's connection settings (from a teammate, never from git) in `.env`; see [docs/database.md: Setting up a station](docs/database.md#setting-up-a-station-each-laptop). That guide also covers viewing and editing the data, schema changes, backups, and handing the database over.

The dashboard also needs [Node.js](https://nodejs.org) 18 or newer. Install its packages once:

```bash
cd ui
npm install
```

### Dashboard (backend + frontend)

One command from the project root starts the backend and the dashboard and opens the dashboard in your browser:

```bash
uv run run.py
```

Allow location access when the browser asks, so events are stamped with where the laptop is. Press Ctrl+C to stop both.

Choose and switch cameras from the dashboard. Optional startup-camera and fallback-location settings are in `.env`, created from `.env.example` on the first run. See [docs/running-and-testing.md](docs/running-and-testing.md) for every setting, a step-by-step test of each feature with the expected results, and troubleshooting. API reference: [docs/backend-api.md](docs/backend-api.md).

### Live camera

From the project root, run:

```bash
uv run src/detectors/yolo_pose.py --model models/pose_pipeline.json
```

This opens camera `0`. Press `Q` or `Esc` to stop. YOLO runs at the camera's native resolution by default so people far from the camera are still found; add `--image-size 640` to downscale (about 3x faster on a CPU, but distant people are missed). To select another camera or request a particular resolution:

```bash
uv run src/detectors/yolo_pose.py --model models/pose_pipeline.json --camera 1 --width 1280 --height 720
```

On first use, your operating system may ask for camera access for Terminal or Python.

The pipeline config selects the ONNX YOLO detector and RTMPose models. Both use
automatic CUDA, DirectML, or CPU selection; on a laptop with both built-in graphics
and a separate GPU, DirectML uses the separate (faster) GPU. The dashboard uses this config by
default; set `FALL_MODEL_PATH` in `.env` to use another config. The default detector is
YOLO26m, which needs a GPU to keep up live; on a laptop without one, set
`FALL_MODEL_PATH=models/pose_pipeline_cpu.json` (the smaller YOLO26n). See
[models/models.md](models/models.md) for speeds.
The main pipeline is split across `src/detectors/yolo_detector.py`, `src/trackers/bytetrack_tracker.py`,
and `src/pose_estimators/rtmpose_estimator.py`, and connected by `src/pose_estimators/rtmpose_pipeline.py`.

To use the alternative single-model YOLO-Pose + ByteTrack pipeline, select its
`.pt` weights instead:

```bash
uv run src/detectors/yolo_pose.py --model models/yolo11n-pose.pt
uv run src/detectors/yolo_pose.py --input data/input/YourVideo.mp4 --model models/yolo11n-pose.pt
```

In the dashboard, set `FALL_MODEL_PATH=models/yolo11n-pose.pt` in `.env` to
choose YOLO-Pose. A `.json` pipeline config chooses YOLO detector + ByteTrack +
RTMPose; a `.pt` pose model chooses YOLO-Pose + ByteTrack. Both return the 12
body keypoints expected by downstream fall detection.

### Prerecorded video

Supply your own video. From the project root, run:

```bash
uv run src/detectors/yolo_pose.py --input data/input/YourVideo.mp4 --model models/pose_pipeline.json
```

Outputs are saved as `data/output/YourVideo.mp4` and `data/output/YourVideo.json`.

### Sliding window

`src/utils/sliding_window.py` keeps the last 30 frames of the 12 body keypoints for each tracked person and turns them into a normalized `(30, 36)` window, the input for the temporal classifier. The five facial keypoints (COCO indices 0-4) are discarded. See [docs/features/10-18-2026-sliding-window.md](docs/features/10-18-2026-sliding-window.md).

### Temporal classifier

Run the skeleton pipeline (frame sampler, pose, tracking, window, classifier, gate) over a video and get the motion-signal plots:

```bash
# YOLO detector -> ByteTrack -> RTMPose (default)
uv run src/fall_pipeline.py --input data/input/YourVideo.mp4 --classifier models/fall_gru.pt

# YOLO-Pose -> ByteTrack
uv run src/fall_pipeline.py --input data/input/YourVideo.mp4 --pipeline yolo-pose --classifier models/fall_gru.pt
```

`--pipeline rtmpose` (the default) starts at `pose_estimators/rtmpose_pipeline.py`,
which calls the separate YOLO detector, ByteTrack, and RTMPose estimator.
`--pipeline yolo-pose` starts at `detectors/yolo_pose.py`, which runs YOLO-Pose
with ByteTrack. `--model` can override either pipeline's default model file.
Both return the same 12 body keypoints. The runner prints the selected pipeline
before processing.

Outputs go to `data/output/YourVideo_temporal/`: `signals.mp4` (annotated video with a scrolling signal panel), `track_<id>.png` (full-length plots per person) and `signals.json`. Leave out `--classifier` to get the plots without a model. To (re)train the bootstrap model on synthetic data, or a model on labeled footage:

```bash
uv run scripts/train_temporal.py --synthetic 1500 --arch gru      # ~1 minute on a CPU
uv run scripts/train_temporal.py --manifest data/labels/manifest.json
```

To use the classifier in the dashboard instead of the rule, set `FALL_CLASSIFIER_PATH=models/fall_gru.pt` in `.env`.

### Tests

```bash
uv run pytest tests -v
```

Expected: `109 passed`. To check that the dashboard compiles, run `npm run build` in `ui/`.

### Incomplete

XClip and continue training temporal model

Team Members: Johnathan Miller, Evan Gruhlkey, Malcom Troy, Allen Basil, Benjamin Aleman

Sponsor: Maksym Pryporov - Maksym_Pryporov@oxy.com

The body-only temporal features require a newly trained checkpoint. Leave `FALL_CLASSIFIER_PATH` empty to use the rule detector until retraining is complete. Existing 17-point pose JSON files still load; their facial points are removed at the file boundary.
