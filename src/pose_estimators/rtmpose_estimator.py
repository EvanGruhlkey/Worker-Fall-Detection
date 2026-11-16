"""RTMPose keypoint estimation for detected and tracked people."""

from rtmlib import RTMPose

from pose_estimators.pose_runtime import load_session


class RTMPoseEstimator:
    """Estimate COCO keypoints for each tracked person box."""

    def __init__(self, model_path, device_id=None):
        self.model = RTMPose.__new__(RTMPose)
        self.model.session = load_session(model_path, device_id)
        input_shape = self.model.session.get_inputs()[0].shape
        self.model.model_input_size = (input_shape[3], input_shape[2])
        self.model.mean = (123.675, 116.28, 103.53)
        self.model.std = (58.395, 57.12, 57.375)
        self.model.backend = "onnxruntime"
        self.model.to_openpose = False

    def estimate(self, frame, boxes):
        """Return the 12 body keypoints for each person box, without facial points."""
        if not boxes:
            return []

        coordinates, confidences = self.model(frame, bboxes=boxes)
        return [
            [
                {"x": float(x), "y": float(y), "confidence": float(confidence)}
                for (x, y), confidence in zip(person_points[5:], person_confidence[5:])
            ]
            for person_points, person_confidence in zip(coordinates, confidences)
        ]
