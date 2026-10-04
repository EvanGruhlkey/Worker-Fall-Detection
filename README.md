# Oxy Worker Fall Detection

#### Computer vision safety software to detect worker falls

### YOLO Pose detecting construction workers

![View annotated video](data/output/001.gif)

### Currently Working

Processes prerecorded videos and saves an annotated MP4 plus JSON with bounding boxes, confidence scores, and 17 body keypoints per person.

### Setup

You can create a virtual env if you want.

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run

Supply your own video. From the project root, run:

```bash
python src/yolo_pose.py --input data/input/YourVideo.mp4 --model models/yolo11n-pose.pt
```

Outputs are saved as `data/output/YourVideo.mp4` and `data/output/YourVideo.json`.

### Incomplete

Person tracking across frames, fall classification, and alarms are not implemented yet. <br><br>

Team Members: Johnathan Miller, Evan Gruhlkey, Malcom Troy, Allen Basil, Benjamin Aleman

Sponsor: Maksym Pryporov - Maksym_Pryporov@oxy.com

