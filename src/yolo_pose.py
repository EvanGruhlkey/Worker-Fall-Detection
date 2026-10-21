import argparse
import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# YOLO's COCO pose model predicts 17 points. The first five are facial points
# (nose, eyes and ears), so expose only the 12 body points at indices 5-16.
COCO_KEYPOINT_COUNT = 17
BODY_KEYPOINT_START = 5
BODY_KEYPOINT_COUNT = COCO_KEYPOINT_COUNT - BODY_KEYPOINT_START
KEYPOINT_FORMAT = "COCO-body-12"


def load_pose_model(model_path):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    if model.task != "pose":
        raise ValueError("Model must be a YOLO pose model")
    return model


def process_frame(model, frame, image_size=640):
    """Run pose tracking on one OpenCV frame and return the drawing and data."""
    result = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=image_size,
        verbose=False,
    )[0]

    # Read tracked box fields individually because tracking adds an ID column
    # to the raw box data.
    boxes = result.boxes.xyxy.cpu().tolist()
    confidences = result.boxes.conf.cpu().tolist()
    track_ids = (
        result.boxes.id.int().cpu().tolist()
        if result.boxes.id is not None
        else [None] * len(boxes)
    )

    keypoints = []
    if result.keypoints is not None:
        keypoints = result.keypoints.data.cpu().tolist()

    people = []
    for box, confidence, track_id, person_keypoints in zip(
        boxes, confidences, track_ids, keypoints
    ):
        if len(person_keypoints) != COCO_KEYPOINT_COUNT:
            raise ValueError("Model must output exactly 17 COCO keypoints")

        body_keypoints = person_keypoints[BODY_KEYPOINT_START:]
        people.append(
            {
                "track_id": track_id,
                "bbox": box,
                "confidence": confidence,
                "keypoints": [
                    {"x": point[0], "y": point[1], "confidence": point[2]}
                    for point in body_keypoints
                ],
            }
        )

    # Hide facial points in the annotated output too. Keeping the 17 slots here
    # lets Ultralytics draw its normal body skeleton, but zero confidence keeps
    # points 0-4 and their connecting lines out of the frame.
    if result.keypoints is not None and result.keypoints.data.shape[-1] >= 3:
        plot_keypoints = result.keypoints.data.clone()
        plot_keypoints[:, :BODY_KEYPOINT_START, 2] = 0
        result.keypoints.data = plot_keypoints

    # Draw the pose results without Ultralytics labels, then add a
    # high-contrast tracking ID and person confidence above each box.
    annotated_frame = result.plot(labels=False, conf=False)
    frame_width = annotated_frame.shape[1]
    for box, confidence, track_id in zip(boxes, confidences, track_ids):
        x1, y1 = int(box[0]), int(box[1])
        id_label = f"ID {track_id}  " if track_id is not None else ""
        label = f"{id_label}Conf {confidence:.0%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        text_x = max(0, min(x1, frame_width - text_width - 8))
        text_y = max(text_height + 8, y1 - 8)

        cv2.rectangle(
            annotated_frame,
            (text_x, text_y - text_height - 8),
            (text_x + text_width + 8, text_y + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            annotated_frame,
            label,
            (text_x + 4, text_y - 4),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated_frame, people


def process_camera(camera_index, model_path, image_size=640, width=None, height=None):
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

    window_name = "YOLO Pose - Live Camera (Q or Esc to quit)"
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


def process_video(input_video, model_path, output_video, output_json, image_size=640):
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
        description="Run YOLO Pose on a live camera or prerecorded video."
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
    parser.add_argument("--model", required=True, help="Path to YOLO Pose weights")
    parser.add_argument(
        "--image-size",
        type=int,
        default=640,
        help="YOLO inference image size (default: 640)",
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
        repo_root = Path(__file__).resolve().parents[1]
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
