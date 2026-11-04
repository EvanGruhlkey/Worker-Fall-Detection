"""Live camera demo: YOLO Pose + ByteTrack shown in an OpenCV window.

This is the real-time counterpart of yolo_pose.py. Instead of reading a
prerecorded video and writing files, it reads frames from a camera, runs pose
estimation and ByteTrack tracking on each frame, and displays the annotated
result in a window as it happens.

Press "q" or Esc in the window to quit.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from yolo_pose import DEFAULT_IMAGE_SIZE, parse_image_size, resolve_image_size

WINDOW_NAME = "Worker Fall Detection - Live"


def draw_track_ids(frame, boxes, track_ids):
    """Draw a high-contrast 'ID N' label above each tracked bounding box."""
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    for box, track_id in zip(boxes, track_ids):
        if track_id is None:
            continue

        x1, y1 = int(box[0]), int(box[1])
        label = f"ID {track_id}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        text_x = max(0, min(x1, width - text_width - 8))
        text_y = max(text_height + 8, y1 - 8)

        cv2.rectangle(
            frame,
            (text_x, text_y - text_height - 8),
            (text_x + text_width + 8, text_y + baseline),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (text_x + 4, text_y - 4),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def draw_status(frame, fps, people_count):
    """Draw a small status bar (FPS and number of tracked people) at the top left."""
    label = f"{fps:5.1f} FPS | {people_count} tracked | q to quit"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, thickness
    )
    cv2.rectangle(
        frame,
        (0, 0),
        (text_width + 16, text_height + baseline + 12),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        label,
        (8, text_height + 6),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def run_live(camera, model_path, image_size=None, output_video=None, mirror=False):
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    if model.task != "pose":
        raise ValueError("Model must be a YOLO pose model")

    # `camera` is either a device index (0, 1, ...) or a stream URL / file path.
    capture = cv2.VideoCapture(camera)
    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open camera {camera!r}. "
            "Check the device index and that no other app is using the camera."
        )

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if output_video is not None:
        output_video = Path(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            camera_fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create video: {output_video}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)

    # Smoothed FPS so the overlay does not flicker.
    fps = 0.0
    last_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera stream ended or frame could not be read.", file=sys.stderr)
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            # persist=True keeps ByteTrack state between frames so IDs stay stable.
            result = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=resolve_image_size(image_size, frame),
                verbose=False,
            )[0]

            boxes = result.boxes.xyxy.cpu().tolist()
            track_ids = (
                result.boxes.id.int().cpu().tolist()
                if result.boxes.id is not None
                else [None] * len(boxes)
            )

            annotated_frame = result.plot(labels=False, conf=False)
            draw_track_ids(annotated_frame, boxes, track_ids)

            now = time.perf_counter()
            instant_fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now
            fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps
            draw_status(annotated_frame, fps, sum(t is not None for t in track_ids))

            cv2.imshow(WINDOW_NAME, annotated_frame)
            if writer is not None:
                writer.write(annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # q or Esc
                break
            # Window closed with the X button.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO Pose + ByteTrack on a live camera feed in an OpenCV window."
    )
    parser.add_argument(
        "--camera",
        default="0",
        help="Camera device index (default 0), or a stream URL / video path",
    )
    parser.add_argument("--model", required=True, help="Path to YOLO Pose weights")
    parser.add_argument(
        "--imgsz",
        default=DEFAULT_IMAGE_SIZE,
        help="Inference size: 'native' (default, no downscaling) or an integer such as 640",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Optional path to also record the annotated feed as an MP4",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Flip the feed horizontally (useful for a laptop webcam demo)",
    )
    args = parser.parse_args()
    try:
        args.imgsz = parse_image_size(args.imgsz)
    except ValueError as error:
        parser.error(f"--imgsz: {error}")
    return args


def main():
    args = parse_args()
    # A plain integer means a device index; anything else is treated as a URL/path.
    camera = int(args.camera) if args.camera.isdigit() else args.camera

    try:
        run_live(
            camera,
            args.model,
            image_size=args.imgsz,
            output_video=args.save,
            mirror=args.mirror,
        )
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if args.save:
        print(f"Saved video: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
