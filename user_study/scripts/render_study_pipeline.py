"""Render verified pipeline overlays for automated study playback only."""
import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from fall_pipeline import TemporalRun
from temporal.pose_io import load_pose_json
from prepare_study_rig_wall import assemble, probe

EDGES = ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7), (6, 8), (8, 10), (7, 9), (9, 11))


def render_source(segment, classifier, cache, ffmpeg):
    source = Path(segment["source"])
    pose_path = Path(segment["pose_source"])
    poses_raw = json.loads(pose_path.read_text(encoding="utf-8"))
    metadata = poses_raw["metadata"]
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if metadata.get("source_sha256") != source_hash:
        raise ValueError(f"Poses do not match source video: {source}")
    output = cache / f"{source.stem}.mp4"
    stamp = output.with_suffix(".json")
    fingerprint = {"video": source_hash, "poses": hashlib.sha256(pose_path.read_bytes()).hexdigest(),
                   "classifier": hashlib.sha256(classifier.read_bytes()).hexdigest()}
    if output.exists() and stamp.exists() and json.loads(stamp.read_text()) == fingerprint:
        return output
    poses = load_pose_json(pose_path).resample(12)
    normalized = Path(segment["signals"]).resolve().parents[2] / "normalized" / source.name
    capture = cv2.VideoCapture(str(normalized))
    process = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "640x360", "-r", "12", "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-g", "24", "-pix_fmt", "yuv420p", str(output)], stdin=subprocess.PIPE)
    state = TemporalRun(12, str(classifier), None, 5, 30)
    scale = min(640 / metadata["width"], 360 / metadata["height"])
    dx, dy = (640 - metadata["width"] * scale) / 2, (360 - metadata["height"] * scale) / 2
    alert_until = {}
    written = 0
    try:
        for index, (timestamp, people) in enumerate(zip(poses.times, poses.people)):
            ok, frame = capture.read()
            if not ok:
                break
            status = state.step(index, timestamp, people)
            for person in people:
                track = person["track_id"]
                probability, above, fired = status.get(track, (float("nan"), False, False))
                if fired:
                    alert_until[track] = timestamp + 1.5
                active = timestamp < alert_until.get(track, -1)
                color = (65, 70, 245) if active else (70, 215, 100)
                x1, y1, x2, y2 = person["bbox"]
                box = [round(x1 * scale + dx), round(y1 * scale + dy), round(x2 * scale + dx), round(y2 * scale + dy)]
                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), color, 2)
                points = [(round(k["x"] * scale + dx), round(k["y"] * scale + dy)) if k["confidence"] >= .3 else None for k in person["keypoints"]]
                for a, b in EDGES:
                    if points[a] and points[b]:
                        cv2.line(frame, points[a], points[b], (235, 210, 65), 2, cv2.LINE_AA)
                for point in points:
                    if point:
                        cv2.circle(frame, point, 2, (40, 240, 240), -1)
                label = f'ID {track} ' + (f'p={probability:.2f}' if math.isfinite(probability) else 'warming') + (' ALERT' if active else '')
                cv2.putText(frame, label, (max(0, box[0]), max(16, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 338), (640, 360), (15, 20, 25), -1)
            cv2.putText(frame, 'YOLO / ByteTrack / RTMPose / GRU', (8, 353), cv2.FONT_HERSHEY_SIMPLEX, .4, (220, 225, 230), 1, cv2.LINE_AA)
            process.stdin.write(frame.tobytes())
            written += 1
    finally:
        capture.release(); process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f"Overlay encoder failed: {source}")
    expected = json.loads(Path(segment["signals"]).read_text())["events"]
    expected = [event for event in expected if event["time"] < written / 12]
    if state.events != expected:
        raise ValueError(f"Overlay decisions differ from scheduled detector output: {source}")
    stamp.write_text(json.dumps(fingerprint), encoding="utf-8")
    print(f"Annotated {source.stem}", flush=True)
    return output


def prepare(directory, classifier):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cache = directory / "preparation/pipeline-sources"
    cache.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((directory / "preparation/provenance.json").read_text())
    unique = {segment["source"]: segment for segment in provenance}
    rendered = {source: render_source(segment, classifier, cache, ffmpeg) for source, segment in unique.items()}
    manifest_path = directory / "rig-wall-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for video_set in manifest["sets"]:
        for feed in video_set["feeds"]:
            timeline = [s for s in provenance if s["set"] == video_set["id"] and s["camera"] == feed["id"]]
            output = directory / "media" / f'rig_{video_set["id"]}_{feed["id"]}_pipeline.mp4'
            schedule = [{"clip": {"normalized": rendered[s["source"]], "frames": round(probe(rendered[s["source"]]) * 12)},
                         "frames": round(s["duration"] * 12)} for s in timeline]
            assemble(schedule, output, cache, ffmpeg)
            feed["pipeline_file"] = output.name
        output = directory / "media" / f'rig_{video_set["id"]}_pipeline_wall.mp4'
        command = [ffmpeg, "-y", "-loglevel", "error"]
        for feed in video_set["feeds"]:
            command.extend(["-threads", "1", "-i", str(directory / "media" / feed["pipeline_file"])])
        command.extend(["-filter_complex", "".join(f"[{i}:v]" for i in range(9)) + "xstack=inputs=9:layout=0_0|640_0|1280_0|0_360|640_360|1280_360|0_720|640_720|1280_720[v]", "-map", "[v]", "-t", str(manifest["duration"]), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-threads", "4", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)])
        subprocess.run(command, check=True)
        video_set["pipeline_wall_file"] = output.name
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Pipeline camera views ready", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("data/study/rig-wall-two"))
    parser.add_argument("--classifier", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.directory, args.classifier)
