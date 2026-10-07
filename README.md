# Oxy Worker Fall Detection

#### Computer vision safety software to detect worker falls

### YOLO Pose detecting construction workers

![View annotated video](data/output/001.gif)

### Currently Working

Runs pose detection and person tracking on either a live OpenCV camera feed or a prerecorded video. The live view shows skeletons, tracking IDs, FPS, and the number of detected people.

### Setup

You can create a virtual env if you want.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Live camera

From the project root, run:

```bash
python src/yolo_pose.py --model models/yolo11n-pose.pt
```

This opens camera `0`. Press `Q` or `Esc` to stop. To select another camera or request a particular resolution:

```bash
python src/yolo_pose.py --model models/yolo11n-pose.pt --camera 1 --width 1280 --height 720
```

On first use, your operating system may ask for camera access for Terminal or Python.

### Prerecorded video

Supply your own video. From the project root, run:

```bash
python src/yolo_pose.py --input data/input/YourVideo.mp4 --model models/yolo11n-pose.pt
```

Outputs are saved as `data/output/YourVideo.mp4` and `data/output/YourVideo.json`.

### Incomplete

Fall classification and alarms are not implemented yet. <br><br>

Team Members: Johnathan Miller, Evan Gruhlkey, Malcom Troy, Allen Basil, Benjamin Aleman

Sponsor: Maksym Pryporov - Maksym_Pryporov@oxy.com
