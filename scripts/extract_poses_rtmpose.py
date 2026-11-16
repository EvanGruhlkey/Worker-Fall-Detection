"""YOLO26 + ByteTrack + RTMPose-t pose extraction, writing the rig pose JSON format.

A re-implementation of the team's batch pipeline (whose code is not in the
repo), matched to what its output files record and checked against them: YOLO26n person detection at
640 px, confidence 0.1, NMS IoU 0.7, ByteTrack (Ultralytics), and RTMPose-t
(body7, 256x192, ONNX, SHA-256 a6c2f6a3...) top-down on every tracked box.
`scripts/compare_pose_files.py` checks it against the team's files: on four
rig clips, matched keypoints differ by a median 0.4-0.9% of body height and
keypoint confidences correlate 0.98-0.99. Only borderline detections
(box confidence ~0.2-0.3) differ, in both directions.

    # every frame (use this for videos whose rate is not a multiple of 12, e.g. 20 fps)
    uv run scripts/extract_poses_rtmpose.py data/external/caucafall/videos --out data/output/caucafall
    # sampled like the rig batch (24 fps -> 12)
    uv run scripts/extract_poses_rtmpose.py data/input/rig_falls --out OUT --fps 12
    # phone videos filmed upright that decode sideways: turn them upright first
    uv run scripts/extract_poses_rtmpose.py clip.mov --out OUT --rotate cw
    # fix the timing of JSONs extracted before true_timing() existed (no re-extraction)
    uv run scripts/extract_poses_rtmpose.py data/input/man_made_fall_videos --out data/output/man_made_fall_videos --retime-only

Frame times come from the video's own frame timestamps (true_timing), not
OpenCV's frame-rate guess: phone videos trimmed on the phone keep one stray
frame from before the trim, which makes OpenCV report e.g. "20 fps over 30 s"
for a 10 s, 60 fps clip. That stray first frame is skipped.

The detector is YOLO26m by default (the live dashboard's detector since
2026-11-15; Trial 1's poses used YOLO26n, --yolo data/external/models/yolo26n.pt).
On an NVIDIA laptop run it with the GPU build of PyTorch, several times faster:

    uv run --no-group cpu --group gpu scripts/extract_poses_rtmpose.py data/input/rig_falls --out OUT --fps 12

Models are downloaded once to data/external/models/ and their SHA-256 is
checked. Output per video: <out>/<video stem>.json (same layout as
data/output/rig_falls/*.json). Existing outputs are skipped unless --force.
"""

import argparse
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data" / "external" / "models"
RTMPOSE_URL = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
               "rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip")
RTMPOSE_SHA256 = "a6c2f6a3896a4d51131d14d7a80a3d08b50f559af5a58a45d5b098aef510a70f"
INPUT_W, INPUT_H = 192, 256
MEAN = np.array([123.675, 116.28, 103.53], np.float32)
STD = np.array([58.395, 57.12, 57.375], np.float32)
PADDING = 1.25
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rtmpose_model():
    MODELS.mkdir(parents=True, exist_ok=True)
    onnx = MODELS / "rtmpose-t_body7_256x192.onnx"
    if not onnx.exists():
        archive = MODELS / "rtmpose-t.zip"
        print(f"Downloading RTMPose-t from {RTMPOSE_URL}")
        urllib.request.urlretrieve(RTMPOSE_URL, archive)
        with zipfile.ZipFile(archive) as zipped:
            name = next(n for n in zipped.namelist() if n.endswith("end2end.onnx"))
            onnx.write_bytes(zipped.read(name))
        archive.unlink()
    if sha256(onnx) != RTMPOSE_SHA256:
        raise RuntimeError(f"{onnx} does not match the team's RTMPose-t (SHA-256 {RTMPOSE_SHA256[:12]}...)")
    return onnx


class RTMPose:
    """Top-down RTMPose (SimCC) on person boxes, as in mmpose / rtmlib."""

    def __init__(self, path):
        import onnxruntime

        self.session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, image, boxes):
        if not len(boxes):
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        crops, transforms = [], []
        for box in boxes:
            center, scale = _box_center_scale(box)
            matrix = _warp_matrix(center, scale, (INPUT_W, INPUT_H))
            crop = cv2.warpAffine(image, matrix, (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
            # BGR, as the team's pipeline feeds it (matches its keypoints best).
            crop = (crop.astype(np.float32) - MEAN) / STD
            crops.append(crop.transpose(2, 0, 1))
            transforms.append((center, scale))
        simcc_x, simcc_y = self.session.run(None, {self.input_name: np.stack(crops)})
        x_locs, y_locs = simcc_x.argmax(-1), simcc_y.argmax(-1)
        # Keypoint confidence = mean of the x and y SimCC peaks (matches the team's files).
        scores = (simcc_x.max(-1) + simcc_y.max(-1)) / 2
        split = simcc_x.shape[-1] / INPUT_W  # 2.0 for this model
        points = np.stack([x_locs, y_locs], -1).astype(np.float32) / split
        out = np.empty_like(points)
        for i, (center, scale) in enumerate(transforms):
            out[i] = points[i] / np.array([INPUT_W, INPUT_H]) * scale + center - scale / 2
        return out, np.clip(scores, 0.0, 1.0)


def _box_center_scale(box):
    x1, y1, x2, y2 = box
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], np.float32)
    w, h = (x2 - x1) * PADDING, (y2 - y1) * PADDING
    aspect = INPUT_W / INPUT_H
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    return center, np.array([w, h], np.float32)


def _warp_matrix(center, scale, size):
    """Affine map from the (center, scale) box to the model input, no rotation."""
    dst_w, dst_h = size
    src = np.array([center, center + [0, -scale[1] / 2], center + [-scale[0] / 2, -scale[1] / 2]], np.float32)
    dst = np.array([[dst_w / 2, dst_h / 2], [dst_w / 2, 0], [0, 0]], np.float32)
    return cv2.getAffineTransform(src, dst)


def true_timing(video):
    """``(fps, skip)``: the real frame rate from the video's frame timestamps, and
    how many leading frames to skip because a trim cut follows them.

    Reads every frame's presentation time with ffmpeg (edit lists ignored, so
    cuts show up as gaps). The rate is 1 / the median gap. A gap over 5x the
    median within the first 3 frames is a phone trim: the frames before it are
    skipped. A cut later in the video cannot be fixed by skipping, so it raises.
    Returns ``(None, 0)`` when ffmpeg is unavailable.
    """
    import re
    import subprocess

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None, 0
    out = subprocess.run([ffmpeg, "-hide_banner", "-ignore_editlist", "1", "-i", str(video), "-map", "0:v:0",
                          "-vf", "showinfo", "-f", "null", "-"], capture_output=True, text=True).stderr
    times = np.array([float(t) for t in re.findall(r"pts_time:([0-9.]+)", out)])
    if len(times) < 3:
        return None, 0
    gaps = np.diff(times)
    median = float(np.median(gaps))
    cuts = np.flatnonzero(gaps > 5 * median)
    if len(cuts) and cuts.max() >= 3:
        raise RuntimeError(f"{video.name}: frame times jump by {gaps[cuts.max()]:.1f} s at frame {cuts.max() + 1}; "
                           "the video was cut in the middle, trim it into separate clips first")
    skip = int(cuts.max()) + 1 if len(cuts) else 0
    return round(1.0 / median, 3), skip


def retime(video, out_path):
    """Correct an existing every-frame JSON with ``true_timing`` (rate and stray first frames)."""
    data = json.loads(out_path.read_text(encoding="utf-8"))
    meta = data["metadata"]
    if abs(meta["fps"] - meta["source_fps"]) > 1e-6:
        raise RuntimeError(f"{out_path.name}: only every-frame extractions can be retimed")
    fps, skip = true_timing(video)
    if fps is None:
        raise RuntimeError("ffmpeg (imageio-ffmpeg) is needed to read frame times")
    frames = [f for f in data["frames"] if f["source_frame"] >= skip]
    for index, frame in enumerate(frames):
        frame["sample_index"] = index
        frame["timestamp"] = frame["source_frame"] / fps
    old = meta["source_fps"]
    meta.update({"source_fps": fps, "fps": fps, "frame_count": len(frames), "skipped_leading_frames": skip,
                 "retimed_from_fps": old})
    meta["configuration"]["fps"] = fps
    data["frames"] = frames
    out_path.write_text(json.dumps(data), encoding="utf-8")
    return old, fps, skip


ROTATIONS = {"none": None, "cw": cv2.ROTATE_90_CLOCKWISE, "ccw": cv2.ROTATE_90_COUNTERCLOCKWISE,
             "180": cv2.ROTATE_180}


def process_video(video, out_path, detector, pose, target_fps, image_size, confidence, iou, rotate="none",
                  detector_name="YOLO26m"):
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {video}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    fps, skip = true_timing(video)
    if fps is not None:
        source_fps = fps
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if rotate in ("cw", "ccw"):
        width, height = height, width
    step = 1 if not target_fps else max(1, round(source_fps / target_fps))
    if target_fps and abs(source_fps / step - target_fps) > 0.02 * target_fps:
        raise RuntimeError(f"{video.name}: {source_fps:g} fps cannot be sampled evenly to {target_fps:g}; "
                           "run without --fps (every frame) instead")
    detector.predictor = None  # fresh tracker state per video
    frames, index, started = [], 0, time.perf_counter()
    while True:
        ok, image = capture.read()
        if ok and ROTATIONS[rotate] is not None:
            # Poses must be measured with "down" pointing down: hip heights, drops
            # and torso angles all assume it.
            image = cv2.rotate(image, ROTATIONS[rotate])
        if not ok:
            break
        if index >= skip and (index - skip) % step == 0:
            result = detector.track(image, persist=True, tracker="bytetrack.yaml", imgsz=image_size,
                                    conf=confidence, iou=iou, classes=[0], verbose=False)[0]
            people = []
            if result.boxes is not None and result.boxes.id is not None and len(result.boxes):
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                confs = result.boxes.conf.cpu().numpy()
                points, scores = pose(image, boxes)
                for box, track_id, box_conf, kps, kp_scores in zip(boxes, ids, confs, points, scores):
                    people.append({
                        "track_id": int(track_id),
                        "bbox": [float(v) for v in box],
                        "confidence": float(box_conf),
                        "keypoints": [{"x": float(x), "y": float(y), "confidence": float(s)}
                                      for (x, y), s in zip(kps, kp_scores)],
                    })
            frames.append({"sample_index": len(frames), "source_frame": index,
                           "timestamp": index / source_fps, "people": people, "alerts": []})
        index += 1
    capture.release()
    elapsed = time.perf_counter() - started
    data = {
        "metadata": {
            "source": str(video), "source_sha256": sha256(video), "source_fps": source_fps,
            "source_frames": total, "fps": source_fps / step, "width": width, "height": height,
            "skipped_leading_frames": skip,
            "frame_count": len(frames),
            "configuration": {
                "pipeline": f"{detector_name} + ByteTrack + RTMPose-t", "implementation": "scripts/extract_poses_rtmpose.py",
                "person_class": 0, "image_size": image_size, "confidence": confidence, "nms_iou": iou, "rotate": rotate,
                "fps": source_fps / step, "rtmpose_sha256": RTMPOSE_SHA256, "fall_alerts": False,
            },
            "processing_seconds": elapsed,
        },
        "frames": frames,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data), encoding="utf-8")
    return len(frames), elapsed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("inputs", nargs="+", help="Video files or folders of videos")
    parser.add_argument("--out", required=True, help="Folder for the JSON files")
    parser.add_argument("--fps", type=float, default=None,
                        help="Keep every n-th frame to reach this rate (default: every frame)")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--rotate", choices=sorted(ROTATIONS), default="none",
                        help="Rotate every frame first (cw / ccw / 180), for phone videos that decode sideways")
    parser.add_argument("--yolo", default=str(MODELS / "yolo26m.pt"),
                        help="YOLO weights (yolo26m.pt by default; downloaded if missing)")
    parser.add_argument("--force", action="store_true", help="Redo videos that already have a JSON")
    parser.add_argument("--retime-only", action="store_true",
                        help="Only correct frame times of existing JSONs from the video's own timestamps")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    videos = []
    for item in map(Path, args.inputs):
        videos += sorted(p for p in item.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES) if item.is_dir() else [item]
    out = Path(args.out)
    if args.retime_only:
        for video in videos:
            target = out / f"{video.stem}.json"
            if not target.exists():
                print(f"  {video.name}: no JSON yet, skipped")
                continue
            old, fps, skip = retime(video, target)
            print(f"  {video.name}: {old:g} -> {fps:g} fps" + (f", skipped {skip} stray leading frame(s)" if skip else ""))
        return 0
    todo = [v for v in videos if args.force or not (out / f"{v.stem}.json").exists()]
    print(f"{len(videos)} videos, {len(todo)} to process -> {out}")
    if not todo:
        return 0
    MODELS.mkdir(parents=True, exist_ok=True)
    yolo_path = Path(args.yolo)
    detector = YOLO(str(yolo_path) if yolo_path.exists() else yolo_path.name)
    if not yolo_path.exists() and Path(yolo_path.name).exists():
        Path(yolo_path.name).replace(yolo_path)
    detector_name = "YOLO" + yolo_path.stem[4:] if yolo_path.stem.lower().startswith("yolo") else yolo_path.stem
    import torch
    print(f"Detector {yolo_path.name} on {'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    pose = RTMPose(rtmpose_model())
    failures = []
    for number, video in enumerate(todo, 1):
        try:
            count, elapsed = process_video(video, out / f"{video.stem}.json", detector, pose,
                                           args.fps, args.image_size, args.conf, args.iou, args.rotate,
                                           detector_name)
            print(f"  [{number}/{len(todo)}] {video.name}: {count} frames, {elapsed:.0f}s", flush=True)
        except Exception as error:  # noqa: BLE001 - report and continue with the next video
            failures.append(f"{video.name}: {error}")
            print(f"  [{number}/{len(todo)}] {video.name}: FAILED {error}", flush=True)
    if failures:
        print(f"{len(failures)} failed:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
