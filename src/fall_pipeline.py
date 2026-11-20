"""Batched run of stages 2 to 7 over a recorded video, with signal plots.

Reads a video (or a pose JSON that ``yolo_pose.py`` already produced), drops
frames to the classifier's rate, runs YOLO pose + ByteTrack, keeps a sliding
window per person, scores every full window with the temporal classifier,
applies the threshold gate, and records the motion signals the classifier
sees (hip height, torso angle, vertical velocity) for every person.

    uv run src/fall_pipeline.py --input data/input/test.mp4 --model models/pose_pipeline.json
    uv run src/fall_pipeline.py --input data/input/test.mp4 --model models/yolo11n-pose.pt
    uv run src/fall_pipeline.py --pose-json data/output/001.json --classifier models/fall_gru.pt

Without ``--classifier`` the signals are still computed and plotted, so the
visualisation works before a model exists. Outputs land in
``data/output/<name>_temporal/``:

    signals.mp4      annotated video with a scrolling signal panel beside it
    track_<id>.png   full-length plots for every track seen long enough
    signals.json     every track's per-frame signals, probabilities and events
"""

import argparse
from functools import partial
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.media import VideoWriter  # noqa: E402
from utils.sliding_window import TrackWindows  # noqa: E402
from temporal.detector import drop_duplicate_people, handoff_lost_tracks, score_tracks  # noqa: E402
from temporal.features import window_drop, window_signals  # noqa: E402
from temporal.gate import ThresholdGate  # noqa: E402
from temporal.model import load_checkpoint  # noqa: E402
from temporal.pose_io import load_pose_json  # noqa: E402
from temporal.sampler import FrameSampler  # noqa: E402
from temporal.visualize import (  # noqa: E402
    SignalPanel,
    TrackSignals,
    compose,
    draw_track_status,
    plot_track_timeline,
)
from detectors.yolo_pose import (  # noqa: E402
    DEFAULT_IMAGE_SIZE,
    parse_image_size,
)
PANEL_WIDTH, PANEL_HEIGHT = 640, 720
# Sampling rate when no classifier says otherwise; matches the rig pose files.
DEFAULT_TARGET_FPS = 12.0


def _video_frames(path, max_frames=None):
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = video.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    def frames():
        try:
            index = 0
            while max_frames is None or index < max_frames:
                ok, frame = video.read()
                if not ok:
                    break
                yield index, frame
                index += 1
        finally:
            video.release()

    return fps, total, frames()


def _json_frames(path, max_frames=None):
    """Frames of a pose JSON from either pipeline, numbered 0, 1, 2, ...

    Returns ``(fps, count, frames, start_seconds)``; ``fps`` is the rate of the
    frames in the file, which is already sampled for the RTMPose pipeline.
    """
    poses = load_pose_json(path)
    people = poses.people if max_frames is None else poses.people[:max_frames]
    return poses.fps, len(people), enumerate(people), poses.times[0]


def choose_focus(histories, now, recent_seconds=2.0):
    """The track to show in the panel: the most alarming recently, else the longest."""
    best, best_score = None, None
    for track_id, history in histories.items():
        if history.last_time is None or now - history.last_time > recent_seconds:
            continue
        recent = [p for t, p in zip(history.time, history.probability) if now - t <= recent_seconds]
        peak = max((p for p in recent if not np.isnan(p)), default=-1.0)
        score = (peak, len(history))
        if best_score is None or score > best_score:
            best, best_score = track_id, score
    return best


class TemporalRun:
    """Stages 5 to 7 plus signal bookkeeping for one video."""

    def __init__(self, fps, classifier=None, threshold=None, debounce=None, cooldown_seconds=30.0,
                 window_size=30):
        self.fps = fps
        self.model, self.meta = (None, {})
        if classifier:
            self.model, self.meta = load_checkpoint(classifier)
            window_size = int(self.meta["window_size"])
            if abs(self.meta["fps"] - fps) > 0.35 * self.meta["fps"]:
                print(
                    f"Warning: classifier trained at {self.meta['fps']:.1f} fps, "
                    f"video sampled at {fps:.1f} fps; velocities are rescaled but the "
                    f"window spans a different time",
                    file=sys.stderr,
                )
        self.threshold = float(
            self.meta.get("threshold", 0.8) if threshold is None else threshold
        )
        self.min_drop = float(self.meta.get("min_drop", 0.0))
        self.max_gap = int(self.meta.get("max_gap", 5))
        self.merge_duplicates = bool(self.meta.get("merge_duplicates", False))
        self.windows = TrackWindows(window_size=window_size, max_gap=self.max_gap)
        debounce = int(self.meta.get("debounce", 5) if debounce is None else debounce)
        self.gate = ThresholdGate(self.threshold, debounce, cooldown_seconds)
        self.histories = {}
        self.events = []

    def step(self, index, time_seconds, people):
        """Process one kept frame. Returns {track_id: (probability, above, fired)}."""
        if self.merge_duplicates:
            people = drop_duplicate_people(people)
        for old_id, new_id in handoff_lost_tracks(self.windows, people, index):
            self.gate.rename(old_id, new_id)
            if old_id in self.histories:
                history = self.histories.pop(old_id)
                history.rename(new_id)
                self.histories[new_id] = history
        updated = self.windows.update(index, people)
        scored = score_tracks(self.model, self.windows, updated, self.fps) if self.model else {}

        status = {}
        for track_id in updated:
            window = self.windows.window(track_id)
            signals = window_signals(window, self.fps)
            probability = scored[track_id][0] if track_id in scored else np.nan
            above = fired = False
            if track_id in scored:
                checked = probability
                if self.min_drop > 0 and window_drop(window, self.fps) < self.min_drop:
                    checked = 0.0  # fast-drop check: no fast body drop, no alert
                fired = self.gate.update(track_id, checked, time_seconds)
                above = self.gate.is_above(track_id)
            history = self.histories.setdefault(track_id, TrackSignals(track_id))
            history.append(time_seconds, index, signals, window.box_features[-1, 0], probability, above)
            if fired:
                history.events.append(time_seconds)
                self.events.append(
                    {"track_id": track_id, "time": round(time_seconds, 3), "frame": index,
                     "probability": round(probability, 4)}
                )
            status[track_id] = (probability, above, fired)
        return status


def run(args):
    # create output folder for temporal model, in the future we should get rid of this and send it straight to the temp model
    name = Path(args.pose_json or args.input).stem
    output_dir = Path(args.output_dir) / f"{name}_temporal"
    output_dir.mkdir(parents=True, exist_ok=True)

    start_seconds = 0.0
    if args.pose_json:
        source_fps, total, frames, start_seconds = _json_frames(args.pose_json, args.max_frames)
        pose_model = None
        process_pose_frame = None
        pipeline_name = "saved pose JSON"
    else:
        source_fps, total, frames = _video_frames(args.input, args.max_frames)
        if args.pipeline == "rtmpose":
            from pose_estimators.rtmpose_pipeline import RTMPosePipeline

            pose_model = RTMPosePipeline(args.model)
            process_pose_frame = pose_model.process_frame
            pipeline_name = "YOLO detector + ByteTrack + RTMPose"
        else:
            from detectors.yolo_pose import (
                load_yolo_pose_model,
                process_yolo_pose_frame,
            )

            pose_model = load_yolo_pose_model(args.model)
            process_pose_frame = partial(process_yolo_pose_frame, pose_model)
            pipeline_name = "YOLO-Pose + ByteTrack"

    target_fps = args.target_fps
    if target_fps is None:
        # Sample at the rate the classifier was trained at, so its window spans
        # the same time live and offline.
        target_fps = load_checkpoint(args.classifier)[1]["fps"] if args.classifier else DEFAULT_TARGET_FPS
    sampler = FrameSampler(source_fps, target_fps)
    run_state = TemporalRun(sampler.fps, args.classifier, args.threshold, args.debounce, args.cooldown)
    print(
        f"{name}: {source_fps:.1f} fps source, keeping every {sampler.stride} frame(s) "
        f"-> {sampler.fps:.1f} fps; pipeline: "
        f"{pipeline_name if args.input else 'saved pose JSON'}; "
        f"classifier: {args.classifier or 'none (signals only)'}"
    )

    writer = panel = None
    annotated_video = getattr(args, 'annotated_video', None)
    replay = cv2.VideoCapture(str(annotated_video)) if annotated_video else None
    if replay is not None and not replay.isOpened():
        raise RuntimeError(f'Could not open annotated video: {annotated_video}')
    if (pose_model is not None or replay is not None) and not args.no_video:
        panel = SignalPanel(PANEL_WIDTH, PANEL_HEIGHT, args.panel_seconds, run_state.threshold)
    started = time.perf_counter()
    kept = 0
    for frame_index, payload in frames:
        if not sampler.keep(frame_index):
            continue
        index = sampler.processed_index(frame_index)
        time_seconds = start_seconds + frame_index / source_fps

        if pose_model is not None:
            annotated, people = process_pose_frame(payload, args.image_size)
        else:
            annotated, people = None, payload
            if replay is not None:
                if int(replay.get(cv2.CAP_PROP_POS_FRAMES)) != frame_index:
                    replay.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, annotated = replay.read()
                if not ok:
                    raise RuntimeError(f'Annotated video ended before pose frame {frame_index}')

        status = run_state.step(index, time_seconds, people)
        kept += 1

        if panel is not None:
            if pose_model is not None:
                for person in people:
                    probability, above, fired = status.get(person["track_id"], (np.nan, False, False))
                    draw_track_status(annotated, person, probability, above, fired)
            focus = args.focus_track if args.focus_track is not None else choose_focus(run_state.histories, time_seconds)
            label = run_state.histories[focus].label() if focus is not None else "No tracked person"
            image = compose(annotated, panel.render(run_state.histories.get(focus), time_seconds, label))
            if writer is None:
                writer = VideoWriter(output_dir / "signals.mp4", image.shape[1], image.shape[0], sampler.fps)
            writer.write(image)

        if kept % 25 == 0:
            progress = f"{frame_index + 1}/{total}" if total else str(frame_index + 1)
            print(f"  frame {progress}  tracks {len(run_state.windows)}  events {len(run_state.events)}  "
                  f"({time.perf_counter() - started:.0f}s)")
    if writer is not None:
        writer.close()
    if replay is not None:
        replay.release()

    plots = []
    for track_id, history in sorted(run_state.histories.items()):
        if len(history) < args.min_plot_frames:
            continue
        plots.append(plot_track_timeline(history, output_dir / f"track_{track_id}.png", run_state.threshold,
                                         f"{name}: {history.label()}"))

    summary = {
        "source": str(args.pose_json or args.input),
        "classifier": args.classifier,
        "source_fps": source_fps,
        "processed_fps": sampler.fps,
        "frames_processed": kept,
        "threshold": run_state.threshold,
        "events": run_state.events,
        "tracks": [h.to_dict() for h in run_state.histories.values()],
    }
    (output_dir / "signals.json").write_text(json.dumps(summary), encoding="utf-8")

    print(f"\nProcessed {kept} frames in {time.perf_counter() - started:.0f}s; {len(run_state.histories)} tracks")
    print(f"{'track':>6} {'frames':>7} {'max p':>7}  events")
    for track_id, history in sorted(run_state.histories.items()):
        peak = history.max_probability()
        print(f"{track_id:>6} {len(history):>7} {'-' if peak is None else f'{peak:7.3f}'}  "
              f"{', '.join(f'{t:.1f}s' for t in history.events) or '-'}")
    print(f"\nOutputs in {output_dir}")
    if writer is not None:
        print(f"  signals.mp4")
    for plot in plots:
        print(f"  {Path(plot).name}")
    print("  signals.json")
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Video to analyse")
    source.add_argument("--pose-json", help="Replay a pose JSON from yolo_pose.py instead of running YOLO")
    parser.add_argument(
        "--pipeline",
        choices=("rtmpose", "yolo-pose"),
        help="Choose the ONNX YOLO detector + RTMPose pipeline or YOLO-Pose model",
    )
    parser.add_argument(
        "--model",
        help="Pipeline config (.json) or YOLO-Pose weights (.pt); defaults by --pipeline",
    )
    parser.add_argument('--annotated-video', help='Existing annotated video paired with --pose-json for the graph panel')
    parser.add_argument("--classifier", help="Temporal classifier checkpoint (.pt); omit to plot signals only")
    parser.add_argument(
        "--target-fps", type=float, default=None,
        help=f"Frame rate after sampling (default: the classifier's training rate, else {DEFAULT_TARGET_FPS:g})",
    )
    parser.add_argument("--threshold", type=float, help="Gate threshold (default: from the checkpoint, else 0.8)")
    parser.add_argument("--debounce", type=int, help="Frames above the threshold before an event (default: from the checkpoint, else 5)")
    parser.add_argument("--cooldown", type=float, default=30.0, help="Seconds between events for one person")
    parser.add_argument(
        "--image-size",
        default=DEFAULT_IMAGE_SIZE,
        help="YOLO inference size: 'native' (default) or an integer such as 640",
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data" / "output"))
    parser.add_argument("--no-video", action="store_true", help="Skip the annotated video (plots and JSON only)")
    parser.add_argument("--panel-seconds", type=float, default=6.0, help="Seconds of history in the scrolling panel")
    parser.add_argument("--focus-track", type=int, help="Always show this track in the panel")
    parser.add_argument("--min-plot-frames", type=int, default=15, help="Skip PNGs for tracks shorter than this")
    parser.add_argument("--max-frames", type=int, help="Stop after this many source frames")
    args = parser.parse_args(argv)
    if args.pipeline is None:
        args.pipeline = (
            "yolo-pose"
            if args.model and Path(args.model).suffix.lower() == ".pt"
            else "rtmpose"
        )
    if args.model is None:
        args.model = str(
            REPO_ROOT
            / "models"
            / (
                "yolo11n-pose.pt"
                if args.pipeline == "yolo-pose"
                else "pose_pipeline.json"
            )
        )
    expected_suffix = ".pt" if args.pipeline == "yolo-pose" else ".json"
    if Path(args.model).suffix.lower() != expected_suffix:
        parser.error(
            f"--pipeline {args.pipeline} requires a {expected_suffix} model file"
        )
    if args.annotated_video and not args.pose_json:
        parser.error('--annotated-video requires --pose-json')
    if args.target_fps is not None and args.target_fps <= 0:
        parser.error("--target-fps must be greater than zero")
    try:
        args.image_size = parse_image_size(args.image_size)
    except ValueError as error:
        parser.error(f"--image-size: {error}")
    if args.threshold is not None and not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    return args


def main():
    args = parse_args()
    try:
        return run(args)
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
