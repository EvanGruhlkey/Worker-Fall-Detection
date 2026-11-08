"""Run the ONNX YOLO detector, ByteTrack, and RTMPose pipeline."""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

# Allow running this file directly from the repository root.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_IMAGE_SIZE = 640
KEYPOINT_FORMAT = "COCO-body-12"
COCO_KEYPOINT_COUNT = 17
FACE_KEYPOINT_COUNT = 5


def load_yolo_pose_model(model_path):
    """Load a single-model Ultralytics YOLO-Pose model and ByteTrack weights."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model not found: {model_path}")

    if model_path.suffix.lower() != ".pt":
        raise ValueError("YOLO-Pose weights must be a .pt file")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if model.task != "pose":
        raise ValueError("The .pt model must be a YOLO pose model")
    return model


def process_yolo_pose_frame(model, frame, image_size=DEFAULT_IMAGE_SIZE):
    """Run one frame through YOLO-Pose + ByteTrack and return 12 body keypoints."""
    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=image_size,
        verbose=False,
    )[0]
    boxes = result.boxes.xyxy.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    track_ids = (
        result.boxes.id.int().cpu().tolist()
        if result.boxes.id is not None
        else [None] * len(boxes)
    )
    if len(confidences) != len(boxes) or len(track_ids) != len(boxes):
        raise ValueError("YOLO-Pose returned inconsistent boxes, confidences, and track IDs")

    keypoints = (
        result.keypoints.data.cpu().tolist()
        if result.keypoints is not None
        else []
    )
    if len(keypoints) != len(boxes):
        if boxes:
            raise ValueError("YOLO-Pose must return keypoints for every detected person")
        keypoints = []

    people = []
    for box, confidence, track_id, person_keypoints in zip(
        boxes, confidences, track_ids, keypoints
    ):
        if len(person_keypoints) != COCO_KEYPOINT_COUNT:
            raise ValueError("YOLO-Pose must output exactly 17 COCO keypoints")
        people.append(
            {
                "track_id": track_id,
                "bbox": box,
                "confidence": confidence,
                "keypoints": [
                    {
                        "x": point[0],
                        "y": point[1],
                        "confidence": point[2],
                    }
                    for point in person_keypoints[FACE_KEYPOINT_COUNT:]
                ],
            }
        )

    annotated_frame = result.plot(labels=False, conf=False)
    for box, track_id in zip(boxes, track_ids):
        if track_id is None:
            continue
        x1, y1 = int(box[0]), int(box[1])
        cv2.putText(
            annotated_frame,
            f"ID {track_id}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated_frame, people


def load_pose_model(model_path):
    """Load either the RTMPose pipeline config or YOLO-Pose weights."""
    model_path = Path(model_path)
    if model_path.suffix.lower() == ".json":
        if not model_path.is_file():
            raise FileNotFoundError(f"Pose pipeline config not found: {model_path}")
        from pose_estimators.rtmpose_pipeline import RTMPosePipeline

        return RTMPosePipeline(model_path)
    return load_yolo_pose_model(model_path)


def process_frame(model, frame, image_size=DEFAULT_IMAGE_SIZE):
    """Run one frame through the selected backend and return body keypoints."""
    pipeline_processor = getattr(model, "process_frame", None)
    if callable(pipeline_processor):
        return pipeline_processor(frame, image_size)
    return process_yolo_pose_frame(model, frame, image_size)


def process_camera(
    camera_index,
    model_path,
    image_size=DEFAULT_IMAGE_SIZE,
    width=None,
    height=None,
):
    """Display pose tracking from a live OpenCV camera feed."""
    model = load_pose_model(model_path)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Could not open camera {camera_index}. Check the camera index and permissions."
        )

    if width is not None:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    window_name = "Pose Tracking - Live Camera (Q or Esc to quit)"
    smoothed_fps = 0.0
    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError(f"Could not read a frame from camera {camera_index}")

            annotated_frame, people = process_frame(model, frame, image_size)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time
            instantaneous_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            smoothed_fps = (
                instantaneous_fps
                if smoothed_fps == 0.0
                else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
            )

            status = f"FPS {smoothed_fps:.1f}  People {len(people)}"
            cv2.putText(
                annotated_frame,
                status,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            # Closing the window should stop the loop too.
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def process_video(
    input_video,
    model_path,
    output_video,
    output_json,
    image_size=DEFAULT_IMAGE_SIZE,
):
    """Process a video and save annotated frames plus per-frame pose data."""
    input_video = Path(input_video)
    output_video = Path(output_video)
    output_json = Path(output_json)

    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    model = load_pose_model(model_path)
    video = cv2.VideoCapture(str(input_video))
    if not video.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        video.release()
        raise RuntimeError(f"Could not create video: {output_video}")

    frames = []
    frame_index = 0

    try:
        while True:
            ok, frame = video.read()
            if not ok:
                break

            annotated_frame, people = process_frame(model, frame, image_size)
            writer.write(annotated_frame)
            frames.append({"frame_index": frame_index, "people": people})
            frame_index += 1
    finally:
        video.release()
        writer.release()

    if frame_index == 0:
        raise RuntimeError(f"Video contains no decodable frames: {input_video}")

    data = {
        "metadata": {
            "source": str(input_video),
            "model": str(model_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_index,
            "keypoint_format": KEYPOINT_FORMAT,
        },
        "frames": frames,
    }
    output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run pose tracking on a live camera or prerecorded video."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", help="Path to a prerecorded input video")
    source.add_argument(
        "--camera",
        type=int,
        nargs="?",
        const=0,
        help="Open a live camera (default index: 0)",
    )
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parents[2] / "models" / "pose_pipeline.json"),
        help="JSON config selecting the YOLO and RTMPose ONNX models",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Inference image size (default: {DEFAULT_IMAGE_SIZE})",
    )
    parser.add_argument("--width", type=int, help="Requested live camera width")
    parser.add_argument("--height", type=int, help="Requested live camera height")
    args = parser.parse_args(argv)

    # With no explicit source, use the computer's default camera.
    if args.input is None and args.camera is None:
        args.camera = 0
    if args.camera is not None and args.camera < 0:
        parser.error("--camera must be zero or greater")
    if args.image_size <= 0:
        parser.error("--image-size must be greater than zero")
    if args.width is not None and args.width <= 0:
        parser.error("--width must be greater than zero")
    if args.height is not None and args.height <= 0:
        parser.error("--height must be greater than zero")
    return args


def main():
    args = parse_args()

    try:
        if args.input is None:
            print(f"Opening camera {args.camera}. Press Q or Esc to quit.")
            process_camera(
                args.camera,
                args.model,
                image_size=args.image_size,
                width=args.width,
                height=args.height,
            )
            return 0

        input_video = Path(args.input)
        repo_root = Path(__file__).resolve().parents[2]
        output_dir = repo_root / "data" / "output"
        output_video = output_dir / f"{input_video.stem}.mp4"
        output_json = output_dir / f"{input_video.stem}.json"
        process_video(
            input_video,
            args.model,
            output_video,
            output_json,
            image_size=args.image_size,
        )
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Saved video: {output_video}")
    print(f"Saved JSON:  {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
