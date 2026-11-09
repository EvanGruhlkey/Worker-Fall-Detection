"""Connect YOLO detection, ByteTrack tracking, and RTMPose estimation."""

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator

from detectors.yolo_detector import YoloPersonDetector
from pose_estimators.rtmpose_estimator import RTMPoseEstimator
from trackers.bytetrack_tracker import PersonTracker


class RTMPosePipeline:
    """Run YOLO -> ByteTrack -> RTMPose while exposing a shared frame format."""

    target_fps = 12

    def __init__(self, config_path):
        config_path = Path(config_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("backend", "rtmpose") not in ("rtmpose", "directml_rtmpose"):
            raise ValueError("Expected rtmpose pipeline configuration")

        for key in ("detector", "pose"):
            if Path(config[key]).suffix.lower() != ".onnx":
                raise ValueError(f"{key} must be an ONNX model (.onnx)")

        device_id = config.get("device_id", 0)
        self.detector = YoloPersonDetector(
            (config_path.parent / config["detector"]).resolve(), device_id
        )
        self.tracker = PersonTracker()
        self.pose_estimator = RTMPoseEstimator(
            (config_path.parent / config["pose"]).resolve(), device_id
        )

    def process_frame(self, frame, image_size=640):
        """Return the annotated frame and detected people for one input frame."""
        if image_size != YoloPersonDetector.input_size:
            raise ValueError("The YOLO detector is exported at fixed image size 640")

        boxes, scores = self.detector.detect(frame)
        tracks = self.tracker.update(boxes, scores, frame)
        track_boxes = [track[:4].tolist() for track in tracks]
        keypoints = self.pose_estimator.estimate(frame, track_boxes)
        if len(keypoints) != len(tracks):
            raise ValueError("RTMPose must return keypoints for every tracked person")

        people = []
        for track, joints in zip(tracks, keypoints):
            people.append(
                {
                    "track_id": int(track[4]),
                    "bbox": track[:4].tolist(),
                    "confidence": float(track[5]),
                    "keypoints": joints,
                }
            )

        return self._annotate(frame, people), people

    @staticmethod
    def _annotate(frame, people):
        annotator = Annotator(frame.copy(), line_width=2, pil=False)
        for person in people:
            x1, y1, x2, y2 = map(round, person["bbox"])
            cv2.rectangle(annotator.im, (x1, y1), (x2, y2), (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(
                annotator.im,
                f'ID {person["track_id"]}  Conf {person["confidence"]:.0%}',
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            points = np.array(
                [
                    [joint["x"], joint["y"], joint["confidence"]]
                    for joint in person["keypoints"]
                ],
                dtype=np.float32,
            )
            points = np.vstack((np.zeros((5, 3), dtype=np.float32), points))
            annotator.kpts(
                points, shape=frame.shape[:2], radius=3, kpt_line=True, conf_thres=0.3
            )
        return annotator.result()
