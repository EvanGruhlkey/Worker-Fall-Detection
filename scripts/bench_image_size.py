"""Compare how many people YOLO pose finds at different inference sizes.

Runs the pose model (no tracker, so counts are not affected by track
confirmation) on every ``--stride``-th frame of each video and reports the
mean people per frame, mean box confidence, smallest box found and CPU time.

    uv run scripts/bench_image_size.py "data/input/large_group/**/*.mp4" --sizes native 640
"""

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from detectors.yolo_pose import load_pose_model, parse_image_size, resolve_image_size  # noqa: E402


def bench_video(model, path, sizes, stride):
    video = cv2.VideoCapture(str(path))
    frames = []
    index = 0
    while True:
        ok, frame = video.read()
        if not ok:
            break
        if index % stride == 0:
            frames.append(frame)
        index += 1
    video.release()
    if not frames:
        raise RuntimeError(f"No frames read from {path}")

    results = {}
    for label, size in sizes:
        counts, confidences, heights = [], [], []
        start = time.perf_counter()
        for frame in frames:
            result = model.predict(frame, imgsz=resolve_image_size(size, frame), verbose=False)[0]
            counts.append(len(result.boxes))
            confidences += result.boxes.conf.tolist()
            if len(result.boxes):
                heights += (result.boxes.xyxy[:, 3] - result.boxes.xyxy[:, 1]).tolist()
        results[label] = {
            "mean_people": float(np.mean(counts)),
            "max_people": int(max(counts)),
            "mean_conf": float(np.mean(confidences)) if confidences else 0.0,
            "min_box_height_px": float(min(heights)) if heights else 0.0,
            "ms_per_frame": 1000 * (time.perf_counter() - start) / len(frames),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("videos", help="Glob of videos, e.g. 'data/input/**/*.mp4'")
    parser.add_argument("--model", default=str(REPO_ROOT / "models" / "yolo11n-pose.pt"))
    parser.add_argument("--sizes", nargs="+", default=["native", "640"])
    parser.add_argument("--stride", type=int, default=6, help="Use every N-th frame")
    parser.add_argument("--output", help="Write the results as JSON here")
    args = parser.parse_args()

    sizes = [(label, parse_image_size(label)) for label in args.sizes]
    paths = sorted(glob.glob(args.videos, recursive=True))
    if not paths:
        parser.error(f"no videos match {args.videos!r}")

    model = load_pose_model(args.model)
    report = {}
    for path in paths:
        report[Path(path).name] = bench_video(model, path, sizes, args.stride)
        line = "  ".join(
            f"{label}: {r['mean_people']:.1f} people/frame, {r['ms_per_frame']:.0f} ms"
            for label, r in report[Path(path).name].items()
        )
        print(f"{Path(path).name}: {line}", flush=True)

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
