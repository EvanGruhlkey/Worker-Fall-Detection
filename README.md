# Oxy Worker Fall Detection

#### Computer vision safety software to detect worker falls

### YOLO Pose detecting construction workers

![View annotated video](data/output/001.gif)

### Currently Working

- **Pose detection and tracking** on a live camera or a prerecorded video. The live view shows skeletons, tracking IDs, FPS, and the number of detected people.
- **Backend server** (`src/backend/`). Runs the camera through the models and a placeholder fall detector. Each detected fall is saved to a database with its time, camera, location, confidence, a snapshot and a short clip.
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

This downloads Python 3.12 if needed, creates `.venv`, and installs the locked dependencies. You do not need to activate the virtual env; prefix commands with `uv run` instead.

To add a dependency, use `uv add <package>` (this updates `pyproject.toml` and `uv.lock`). Commit both files.

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

Settings (which camera to use, fallback location) are in `.env`, created from `.env.example` on the first run. See [docs/running-and-testing.md](docs/running-and-testing.md) for every setting, a step-by-step test of each feature with the expected results, and troubleshooting. API reference: [docs/backend-api.md](docs/backend-api.md).

### Live camera

From the project root, run:

```bash
uv run src/yolo_pose.py --model models/yolo11n-pose.pt
```

This opens camera `0`. Press `Q` or `Esc` to stop. To select another camera or request a particular resolution:

```bash
uv run src/yolo_pose.py --model models/yolo11n-pose.pt --camera 1 --width 1280 --height 720
```

On first use, your operating system may ask for camera access for Terminal or Python.

### Prerecorded video

Supply your own video. From the project root, run:

```bash
uv run src/yolo_pose.py --input data/input/YourVideo.mp4 --model models/yolo11n-pose.pt
```

Outputs are saved as `data/output/YourVideo.mp4` and `data/output/YourVideo.json`.

### Sliding window

`src/sliding_window.py` keeps the last 30 frames of the 12 body keypoints for each tracked person and turns them into a normalized `(30, 36)` window, the input for the temporal classifier. The five facial keypoints (COCO indices 0-4) are discarded. See [docs/features/10-18-2026-sliding-window.md](docs/features/10-18-2026-sliding-window.md).

### Tests

```bash
uv run pytest tests -v
```

Expected: `69 passed`. To check that the dashboard compiles, run `npm run build` in `ui/`.

### Incomplete

The temporal classifier and the X-CLIP verifier are not implemented yet. Until the classifier is ready, falls are flagged by a simple rule in `src/backend/fall_detector.py`. The dashboard's filter, export and settings buttons are not connected yet. Alerts appear only on the dashboard (no sound, SMS or email). <br><br>

Team Members: Johnathan Miller, Evan Gruhlkey, Malcom Troy, Allen Basil, Benjamin Aleman

Sponsor: Maksym Pryporov - Maksym_Pryporov@oxy.com
