"""Speed test for the live pose pipeline (detector + ByteTrack + RTMPose-t) on this laptop.

The live system analyzes 12 frames per second from one camera, so every frame
has about 83 ms for detection, tracking and pose on everyone in view. This
script runs each candidate detector over a busy rig clip and reports how fast
the full pipeline runs, so we can pick the most accurate detector that still
keeps up on every team laptop.

    uv run scripts/benchmark_pipeline.py                        # YOLO26n/s/m, PyTorch and ONNX, 640 px
    uv run scripts/benchmark_pipeline.py --models yolo26n yolo26s --formats pt onnx openvino
    uv run scripts/benchmark_pipeline.py --image-sizes 640 960 --frames 60

Each teammate runs it once and commits the result file it writes to
runs/speed/<computer name>.json. Weights and exported models are cached in
data/external/models/ (not committed).
"""

import argparse
import json
import os
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_poses_rtmpose import MODELS, ROOT, RTMPose, rtmpose_model  # noqa: E402

TARGET_FPS = 12.0
DEFAULT_CLIP = ROOT / "data" / "input" / "rig_falls" / "sudden_collapse.mp4"  # ten workers in view
OUT_DIR = ROOT / "runs" / "speed"


def cpu_name():
    if platform.system() == "Windows":
        try:
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:  # noqa: BLE001
            pass
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    return platform.processor() or platform.machine()


def hardware():
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    if gpu is None and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        gpu = "Apple GPU (MPS)"
    return {"computer": socket.gethostname(), "os": f"{platform.system()} {platform.release()}",
            "cpu": cpu_name(), "cpu_threads": os.cpu_count(), "gpu": gpu,
            "torch": torch.__version__, "cuda": torch.cuda.is_available()}


def weights(name):
    """Download <name>.pt once into data/external/models/."""
    from ultralytics import YOLO

    path = MODELS / f"{name}.pt"
    if not path.exists():
        YOLO(f"{name}.pt")  # downloads into the working directory
        shutil.move(f"{name}.pt", path)
    return path


def detector_for(name, fmt, image_size, device):
    """A YOLO model in the requested format, exporting (and caching) it if needed."""
    from ultralytics import YOLO

    pt = weights(name)
    if fmt == "pt":
        return YOLO(str(pt)), str(pt)
    suffix = {"onnx": f"{name}_{image_size}.onnx", "openvino": f"{name}_{image_size}_openvino_model"}[fmt]
    target = MODELS / suffix
    if not target.exists():
        exported = Path(YOLO(str(pt)).export(format=fmt, imgsz=image_size, dynamic=False, verbose=False))
        shutil.move(str(exported), target)
    return YOLO(str(target), task="detect"), str(target)


def read_frames(clip, count):
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {clip}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, round(source_fps / TARGET_FPS))
    frames, index = [], 0
    while len(frames) < count:
        ok, image = capture.read()
        if not ok:
            break
        if index % step == 0:
            frames.append(image)
        index += 1
    capture.release()
    return frames


def run_pipeline(detector, pose, frames, image_size, device, warmup):
    detector.predictor = None  # fresh tracker
    detect_ms, pose_ms, people = [], [], []
    for number, image in enumerate(frames):
        start = time.perf_counter()
        result = detector.track(image, persist=True, tracker="bytetrack.yaml", imgsz=image_size,
                                conf=0.1, iou=0.7, classes=[0], device=device, verbose=False)[0]
        middle = time.perf_counter()
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((0, 4))
        pose(image, boxes)
        end = time.perf_counter()
        if number >= warmup:
            detect_ms.append((middle - start) * 1000)
            pose_ms.append((end - middle) * 1000)
            people.append(len(boxes))
    total = [d + p for d, p in zip(detect_ms, pose_ms)]
    median = statistics.median(total)
    p95 = sorted(total)[int(0.95 * (len(total) - 1))]
    return {"detect_ms": round(statistics.median(detect_ms), 1), "pose_ms": round(statistics.median(pose_ms), 1),
            "total_ms": round(median, 1), "total_p95_ms": round(p95, 1), "fps": round(1000 / median, 1),
            "people_per_frame": round(statistics.mean(people), 1),
            "keeps_up": 1000 / p95 >= TARGET_FPS}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clip", default=str(DEFAULT_CLIP))
    parser.add_argument("--models", nargs="+", default=["yolo26n", "yolo26s", "yolo26m"])
    parser.add_argument("--formats", nargs="+", default=["pt", "onnx"], choices=["pt", "onnx", "openvino"])
    parser.add_argument("--image-sizes", nargs="+", type=int, default=[640])
    parser.add_argument("--frames", type=int, default=72, help="Frames to time at 12 fps (72 = 6 s of video)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default=None, help="cpu, 0 (first NVIDIA GPU), mps; default: best available")
    args = parser.parse_args(argv)

    info = hardware()
    device = args.device or ("0" if info["cuda"] else "cpu")
    print(f"Computer: {info['computer']} | CPU: {info['cpu']} ({info['cpu_threads']} threads) | "
          f"GPU: {info['gpu'] or 'none'} | running on: {device}")
    frames = read_frames(Path(args.clip), args.frames + args.warmup)
    print(f"Clip: {Path(args.clip).name}, {len(frames)} frames at 12 fps. Target: {TARGET_FPS:g} fps "
          f"(under {1000 / TARGET_FPS:.0f} ms per frame, checked on the slowest 5% of frames)\n")
    pose = RTMPose(rtmpose_model())

    results = []
    header = f"{'detector':<10}{'format':<10}{'size':>5}{'detect':>9}{'pose':>8}{'total':>8}{'p95':>8}{'fps':>7}  verdict"
    print(header + "\n" + "-" * len(header))
    for name in args.models:
        for fmt in args.formats:
            if fmt != "pt" and device != "cpu":
                continue  # exported formats here are CPU paths; on a GPU, PyTorch is the fast one
            for size in args.image_sizes:
                try:
                    detector, path = detector_for(name, fmt, size, device)
                    row = run_pipeline(detector, pose, frames, size, device, args.warmup)
                except Exception as error:  # noqa: BLE001 - report and keep going
                    print(f"{name:<10}{fmt:<10}{size:>5}  not available: {str(error).splitlines()[0][:70]}")
                    results.append({"detector": name, "format": fmt, "image_size": size, "error": str(error)[:300]})
                    continue
                row = {"detector": name, "format": fmt, "image_size": size, **row}
                results.append(row)
                verdict = "OK, keeps up" if row["keeps_up"] else "too slow"
                print(f"{name:<10}{fmt:<10}{size:>5}{row['detect_ms']:>7.0f}ms{row['pose_ms']:>6.0f}ms"
                      f"{row['total_ms']:>6.0f}ms{row['total_p95_ms']:>6.0f}ms{row['fps']:>7.1f}  {verdict}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{info['computer']}.json"
    out.write_text(json.dumps({"date": datetime.now().isoformat(timespec="seconds"), "hardware": info,
                               "device": device, "clip": Path(args.clip).name, "frames": len(frames) - args.warmup,
                               "target_fps": TARGET_FPS, "results": results}, indent=2), encoding="utf-8")
    ok = [r for r in results if r.get("keeps_up")]
    print(f"\nFastest-enough choices on this laptop: "
          + (", ".join(f"{r['detector']} {r['format']} {r['image_size']}" for r in ok) or "none"))
    print(f"Saved {out.relative_to(ROOT)}. Please commit this file so the team can compare laptops.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
