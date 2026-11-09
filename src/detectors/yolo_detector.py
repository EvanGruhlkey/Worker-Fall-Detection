"""YOLO person detection for the YOLO + ByteTrack + RTMPose pipeline."""

import cv2
import numpy as np

from pose_estimators.pose_runtime import load_session


class YoloPersonDetector:
    """Find people in a frame and return boxes and detection confidence scores."""

    input_size = 640

    def __init__(self, model_path, device_id=0):
        self.session = load_session(model_path, device_id)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame):
        height, width = frame.shape[:2]
        scale = min(self.input_size / height, self.input_size / width)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        pad_x = (self.input_size - resized_width) // 2
        pad_y = (self.input_size - resized_height) // 2

        image = cv2.resize(frame, (resized_width, resized_height))
        image = cv2.copyMakeBorder(
            image,
            pad_y,
            self.input_size - resized_height - pad_y,
            pad_x,
            self.input_size - resized_width - pad_x,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        tensor = np.ascontiguousarray(
            image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32
        ) / 255
        output = self.session.run(None, {self.input_name: tensor})[0]
        if output.shape != (1, 84, 8400):
            raise ValueError(f"Expected YOLO COCO output (1, 84, 8400); got {output.shape}")

        detections = output[0].T
        people = detections[
            (detections[:, 4] >= 0.1) & (detections[:, 4:84].argmax(axis=1) == 0)
        ]
        if not len(people):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        boxes = people[:, :4].copy()
        boxes[:, :2] -= boxes[:, 2:] / 2
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), people[:, 4].tolist(), 0.1, 0.7)
        if indices is None or len(indices) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)
        indices = np.asarray(indices).reshape(-1).astype(int)
        boxes = boxes[indices]
        scores = people[indices, 4]

        boxes[:, 2:] += boxes[:, :2]
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_x) / scale).clip(0, width)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_y) / scale).clip(0, height)
        return boxes, scores
