"""Make a short synthetic fall video for testing, from YOLO's sample photo.

A person cut out of ultralytics' bus.jpg stands for 2 s, falls and rotates to
lying down in 0.6 s, then lies still for 3 s (15 FPS, 1280x960). It isn't
realistic footage, but the pose model, ByteTrack and the fall detector all
handle it, so it exercises the full pipeline without filming a real fall.

    uv run scripts/make_synthetic_fall.py                 # -> data/samples/synthetic_fall.mp4
    uv run scripts/make_synthetic_fall.py --output other.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from backend.media import VideoWriter  # noqa: E402  H.264, plays in browsers
WIDTH, HEIGHT, FPS = 1280, 960, 15
BACKGROUND = 110  # plain grey


def cut_out_person(model_path):
    """Crop the tallest person YOLO finds in bus.jpg."""
    image = cv2.imread(str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg"))
    boxes = YOLO(str(model_path))(image, verbose=False)[0].boxes.xyxy.cpu().numpy()
    x1, y1, x2, y2 = max(boxes, key=lambda b: b[3] - b[1]).astype(int)
    return image[y1:y2, x1:x2]


def render(person, angle, center_x, center_y):
    """One frame with the person rotated by `angle` degrees around (center_x, center_y)."""
    frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND, np.uint8)
    height, width = person.shape[:2]
    size = max(height, width) * 2
    canvas = np.full((size, size, 3), BACKGROUND, np.uint8)
    mid = size // 2
    canvas[mid - height // 2:mid - height // 2 + height, mid - width // 2:mid - width // 2 + width] = person
    rotation = cv2.getRotationMatrix2D((mid, mid), angle, 1.0)
    rotated = cv2.warpAffine(canvas, rotation, (size, size), borderValue=(BACKGROUND,) * 3)

    x0, y0 = int(center_x - size / 2), int(center_y - size / 2)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + size, WIDTH), min(y0 + size, HEIGHT)
    frame[ya:yb, xa:xb] = rotated[ya - y0:yb - y0, xa - x0:xb - x0]
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "samples" / "synthetic_fall.mp4")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "yolo11n-pose.pt")
    args = parser.parse_args()

    person = cut_out_person(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = VideoWriter(args.output, WIDTH, HEIGHT, FPS)

    standing_y = HEIGHT * 0.45
    lying_y = HEIGHT - person.shape[1] * 0.5 - 20
    for _ in range(FPS * 2):  # standing, 2 s
        writer.write(render(person, 0, WIDTH / 2, standing_y))
    steps = int(FPS * 0.6)
    for step in range(1, steps + 1):  # falling, 0.6 s
        t = step / steps
        writer.write(render(person, 90 * t, WIDTH / 2, standing_y + (lying_y - standing_y) * t))
    for _ in range(FPS * 3):  # lying still, 3 s
        writer.write(render(person, 90, WIDTH / 2, lying_y))
    writer.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
