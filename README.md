# Oxy Worker Fall Detection

#### Computer vision safety software to detect worker falls

### YOLO Pose detecting construction workers

![View annotated video](data/output/001.gif)

### Currently Working

Runs pose detection and person tracking on either a live OpenCV camera feed or a prerecorded video. The live view shows skeletons, tracking IDs, FPS, and the number of detected people.

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

`src/sliding_window.py` keeps the last 30 frames of keypoints for each tracked person and turns them into a normalized `(30, 51)` window, the input for the temporal classifier. See [docs/features/10-18-2026-sliding-window.md](docs/features/10-18-2026-sliding-window.md).

### Tests

```bash
uv run pytest tests -v
```

### Incomplete

The temporal classifier, fall alarms, and the X-CLIP verifier are not implemented yet. <br><br>

Team Members: Johnathan Miller, Evan Gruhlkey, Malcom Troy, Allen Basil, Benjamin Aleman

Sponsor: Maksym Pryporov - Maksym_Pryporov@oxy.com
