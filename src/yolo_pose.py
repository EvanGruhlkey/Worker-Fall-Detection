import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


def process_video(input_video, model_path, output_video, output_json, image_size=640):
    input_video = Path(input_video)
    model_path = Path(model_path)
    output_video = Path(output_video)
    output_json = Path(output_json)

    if not input_video.is_file():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = YOLO(str(model_path))
    if model.task != "pose":
        raise ValueError("Model must be a YOLO pose model")

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

            result = model(frame, imgsz=image_size, verbose=False)[0]

            # Ultralytics draws its own boxes, keypoints, and skeletons
            # labels=False, conf=False to get rid of person and conf like in demo 1
            writer.write(result.plot(labels=False, conf=False))

            # [x1, y1, x2, y2, confidence, class]
            boxes = result.boxes.data.cpu().tolist()

            # 17 [x, y, confidence] keypoints
            keypoints = []
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().tolist()

            people = []
            for box, person_keypoints in zip(boxes, keypoints):
                if len(person_keypoints) != 17:
                    raise ValueError("Model must output exactly 17 COCO keypoints")

                people.append(
                    {
                        "bbox": box[:4],
                        "confidence": box[4],
                        "keypoints": [
                            {"x": point[0], "y": point[1], "confidence": point[2]}
                            for point in person_keypoints
                        ],
                    }
                )

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
            "keypoint_format": "COCO-17",
        },
        "frames": frames,
    }
    output_json.write_text(json.dumps(data, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLO Pose and save an annotated video plus JSON."
    )
    parser.add_argument("--input", required=True, help="Path to the input video")
    parser.add_argument("--model", required=True, help="Path to YOLO Pose weights")
    return parser.parse_args()


def main():
    args = parse_args()
    input_video = Path(args.input)
    # 001.mp4 becomes data/output/001.mp4 and data/output/001.json.
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = repo_root / "data" / "output"
    output_video = output_dir / f"{input_video.stem}.mp4"
    output_json = output_dir / f"{input_video.stem}.json"

    try:
        process_video(input_video, args.model, output_video, output_json)
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Saved video: {output_video}")
    print(f"Saved JSON:  {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
