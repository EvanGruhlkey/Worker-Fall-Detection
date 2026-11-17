"""Build two two-minute, six-camera sets from the project's AI rig clips.

Each set has six labeled fall occurrences spread across six feeds. Activities
cycle between them. Alerts are actual per-source-clip detector outputs, replayed
at each occurrence's offset; classifier state resets at clip boundaries.
"""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg

REPO = Path(__file__).resolve().parents[2]
FPS = 12
DURATION = 120
CAMERA_COUNT = 6


def probe(path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or fps <= 0:
            raise ValueError(f"Unreadable source: {path}")
        return cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    finally:
        cap.release()


def sources():
    with (REPO / "data/labels/clips.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    result = {"fall": [], "normal": []}
    for row in rows:
        if row["quality"] != "keep" or row["folder"] not in {"rig_falls", "normal_rig_work"}:
            continue
        path = REPO / "data/input" / row["folder"] / row["file"]
        if not path.is_file():
            continue
        is_fall = row["has_fall"] == "1"
        onset = float(row["fall_start_s"]) if is_fall and row["fall_start_s"] else None
        if is_fall and onset is None:
            continue
        result["fall" if is_fall else "normal"].append({"id": row["clip_id"], "path": path, "onset": onset})
    if min(len(result["fall"]), len(result["normal"])) < 4:
        raise ValueError("Need at least four labeled fall clips and four normal clips")
    return result


def normalize(source, work, ffmpeg):
    frames = int(probe(source["path"]) * FPS)
    if frames <= 0:
        raise ValueError("Empty source clip")
    output = work / "normalized" / f'{source["id"]}.mp4'
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(source["path"]), "-an", "-vf",
                        "scale=640:360:force_original_aspect_ratio=decrease,pad=640:360:(ow-iw)/2:(oh-ih)/2,fps=12",
                        "-frames:v", str(frames), "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-g", "24", "-pix_fmt", "yuv420p", str(output)], check=True)
    return {**source, "normalized": output, "frames": round(probe(output) * FPS)}


def detect(source, args, work):
    signals = work / "detections" / f'{source["id"]}_temporal' / "signals.json"
    metadata = signals.with_name("study-source.json")
    pose_path = REPO / "data/output" / source["path"].parent.name / f'{source["id"]}.json'
    video_hash = hashlib.sha256(source["path"].read_bytes()).hexdigest()
    saved_pose = json.loads(pose_path.read_text(encoding="utf-8")) if pose_path.is_file() else None
    use_saved_pose = isinstance(saved_pose, dict) and saved_pose.get("metadata", {}).get("source_sha256") == video_hash
    fingerprint = {"video_sha256": video_hash,
                   "classifier_sha256": hashlib.sha256(args.classifier.read_bytes()).hexdigest(),
                   "pose_source": str(pose_path) if use_saved_pose else str(args.pose_model),
                   "pose_sha256": hashlib.sha256((pose_path if use_saved_pose else args.pose_model).read_bytes()).hexdigest()}
    if not signals.exists() or not metadata.exists() or json.loads(metadata.read_text()) != fingerprint:
        if use_saved_pose:
            sys.path.insert(0, str(REPO / "src"))
            from fall_pipeline import TemporalRun
            from temporal.pose_io import load_pose_json
            poses = load_pose_json(pose_path).resample(FPS)
            state = TemporalRun(poses.fps, str(args.classifier), None, 5, 30)
            for index, (timestamp, people) in enumerate(zip(poses.times, poses.people)):
                state.step(index, timestamp, people)
            signals.parent.mkdir(parents=True, exist_ok=True)
            body = {"source": str(pose_path), "classifier": str(args.classifier), "events": state.events,
                    "pose_configuration": saved_pose["metadata"].get("configuration"), "classifier_reset": True}
            signals.write_text(json.dumps(body), encoding="utf-8")
        else:
            subprocess.run([sys.executable, str(REPO / "src/fall_pipeline.py"), "--input", str(source["path"]), "--model", str(args.pose_model),
                            "--classifier", str(args.classifier), "--image-size", "640", "--target-fps", "12", "--no-video", "--output-dir", str(work / "detections")], cwd=REPO, check=True)
        metadata.write_text(json.dumps(fingerprint), encoding="utf-8")
    body = json.loads(signals.read_text(encoding="utf-8"))
    return {**source, "alerts": body["events"], "signals": str(signals), **fingerprint}


def timeline(normal, falls, camera_index, set_index):
    schedule = []
    offset = 0
    normal_index = camera_index + set_index * 5
    target_frames = DURATION * FPS

    def fill(until):
        nonlocal offset, normal_index
        while offset < until:
            clip = normal[normal_index % len(normal)]
            normal_index += 1
            length = min(clip["frames"], until - offset)
            schedule.append({"clip": clip, "start_frame": offset, "frames": length, "kind": "normal"})
            offset += length

    for event_index in range(6):
        if event_index % CAMERA_COUNT != camera_index:
            continue
        clip = falls[(event_index + set_index * 3) % len(falls)]
        onset_frame = round(clip["onset"] * FPS)
        desired_frame = (18 + event_index * 17 + random.Random(482 + set_index * 1000 + event_index).randint(-1, 1)) * FPS
        fill(desired_frame - onset_frame)
        schedule.append({"clip": clip, "start_frame": offset, "frames": clip["frames"], "kind": "fall"})
        offset += clip["frames"]
    fill(target_frames)
    return schedule


def assemble(schedule, output, work, ffmpeg):
    fragments = []
    for i, segment in enumerate(schedule):
        clip = segment["clip"]
        if segment["frames"] == clip["frames"]:
            fragments.append(clip["normalized"])
        else:
            fragment = work / "cuts" / f'{output.stem}_{i}.mp4'
            fragment.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(clip["normalized"]), "-frames:v", str(segment["frames"]),
                            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-g", "24", "-pix_fmt", "yuv420p", str(fragment)], check=True)
            fragments.append(fragment)
    listing = work / f'{output.stem}.txt'
    listing.write_text("\n".join(f"file '{p.resolve().as_posix()}'" for p in fragments), encoding="utf-8")
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-t", str(DURATION),
                    "-c", "copy", "-movflags", "+faststart", str(output)], check=True)
    if abs(probe(output) - DURATION) > .1:
        raise ValueError(f"Assembled feed has incorrect duration: {output}")


def prepare(args):
    work = args.directory / "preparation"
    media = args.directory / "media"
    work.mkdir(parents=True, exist_ok=True)
    media.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    clips = sources()
    cache = args.cache_directory or work
    prepared = {}
    for kind, group in clips.items():
        prepared[kind] = []
        for source in group:
            print(f'Preparing {source["id"]}', flush=True)
            prepared[kind].append(detect(normalize(source, cache, ffmpeg), args, cache))
    sets, provenance = [], []
    for set_index, set_id in enumerate(("A", "B")):
        normal = prepared["normal"]
        falls = prepared["fall"][set_index::2]
        feeds, truth, alerts = [], [], []
        for camera_index in range(CAMERA_COUNT):
            camera = f"camera{camera_index + 1:02}"
            output = media / f"rig_{set_id}_{camera}.mp4"
            schedule = timeline(normal, falls, camera_index, set_index)
            assemble(schedule, output, work, ffmpeg)
            for segment in schedule:
                clip = segment["clip"]
                offset = segment["start_frame"] / FPS
                length = segment["frames"] / FPS
                if segment["kind"] == "fall":
                    truth.append({"camera": camera, "time": offset + clip["onset"]})
                alerts.extend({"camera": camera, "time": round(offset + e["time"], 3)} for e in clip["alerts"] if e["time"] < length)
                provenance.append({"set": set_id, "camera": camera, "offset": offset, "duration": length, "source": str(clip["path"]),
                                   "source_sha256": clip["video_sha256"], "signals": clip["signals"], "pose_source": clip["pose_source"], "pose_sha256": clip["pose_sha256"], "classifier_reset": True})
            feeds.append({"id": camera, "name": f"Camera {camera_index + 1:02}", "file": output.name, "duration": probe(output)})
            print(f"Assembled {set_id} / {camera}: {DURATION} seconds", flush=True)
        sets.append({"id": set_id, "feeds": feeds, "truth": sorted(truth, key=lambda e: e["time"]), "alerts": sorted(alerts, key=lambda e: e["time"])})
    checksum = hashlib.sha256(args.classifier.read_bytes()).hexdigest()
    manifest = {"title": "AI rig footage / six-camera study", "mode": "study", "duration": DURATION,
                "early_tolerance": 1, "late_tolerance": 15, "detector_version": f"Verified saved YOLO26m/RTMPose poses (fallback {args.pose_model.name}); checkpoint SHA256 {checksum}",
                "alert_source": "Actual per-source-clip fall_pipeline outputs, replayed at clip offsets; classifier state resets on clip changes. AI-generated rig footage.", "sets": sets}
    (work / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    path = args.directory / "rig-wall-manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    from build_study_mosaic import prepare as prepare_mosaic
    prepare_mosaic(args.directory)
    print(f"Ready: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=REPO / "data/study/rig-wall-two")
    parser.add_argument("--cache-directory", type=Path)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, default=REPO / "models/pose_pipeline_cpu.json")
    prepare(parser.parse_args())
