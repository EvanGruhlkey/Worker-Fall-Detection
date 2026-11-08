"""Read pose JSON, converting older COCO-17 files to the 12 body points."""

import json
from pathlib import Path


def load_pose_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    metadata = data["metadata"]
    keypoint_format = metadata.get("keypoint_format")
    if keypoint_format not in (None, "COCO-17", "COCO-body-12"):
        raise ValueError(f"Unsupported keypoint format: {keypoint_format!r}")
    expected = {"COCO-17": 17, "COCO-body-12": 12}.get(keypoint_format)
    for frame in data["frames"]:
        for person in frame["people"]:
            points = person["keypoints"]
            if len(points) not in (12, 17) or (expected is not None and len(points) != expected):
                raise ValueError("Pose JSON must contain COCO-17 or COCO-body-12 keypoints")
            if len(points) == 17:
                person["keypoints"] = points[5:]
    metadata["keypoint_format"] = "COCO-body-12"
    return data
