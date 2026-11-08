import importlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

import numpy as np

from detectors import yolo_pose


def test_json_config_loads_rtmpose_pipeline(tmp_path, monkeypatch):
    config = tmp_path / "rtmpose.json"
    config.touch()

    class FakeTracker:
        def __init__(self, path):
            assert path == config

        def process_frame(self, frame, image_size):
            return frame, []

    module = ModuleType("pose_estimators.rtmpose_pipeline")
    module.RTMPosePipeline = FakeTracker
    monkeypatch.setitem(sys.modules, "pose_estimators.rtmpose_pipeline", module)

    assert isinstance(yolo_pose.load_pose_model(config), FakeTracker)


def test_pt_weights_load_yolo_pose_model(tmp_path, monkeypatch):
    weights = tmp_path / "pose.pt"
    weights.touch()
    expected = type("FakeYOLO", (), {"task": "pose"})()
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = lambda path: expected
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    assert yolo_pose.load_yolo_pose_model(weights) is expected


def test_pt_weights_reject_non_pose_model(tmp_path, monkeypatch):
    weights = tmp_path / "detect.pt"
    weights.touch()
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLO = lambda path: type("FakeYOLO", (), {"task": "detect"})()
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    with pytest.raises(ValueError, match="YOLO pose model"):
        yolo_pose.load_yolo_pose_model(weights)


def test_process_frame_runs_yolo_pose_and_returns_body_keypoints():
    class Tensor:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def int(self):
            return self

        def tolist(self):
            return self.value

    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    points = [[[index, index + 1, 0.9] for index in range(17)]]
    boxes = type(
        "Boxes",
        (),
        {
            "xyxy": Tensor([[1, 2, 30, 35]]),
            "conf": Tensor([0.95]),
            "id": Tensor([7]),
        },
    )()
    result = type(
        "Result",
        (),
        {
            "boxes": boxes,
            "keypoints": type("Keypoints", (), {"data": Tensor(points)})(),
            "plot": lambda self, labels, conf: frame.copy(),
        },
    )()

    class FakeModel:
        def track(self, received_frame, **kwargs):
            assert received_frame is frame
            assert kwargs == {
                "persist": True,
                "tracker": "bytetrack.yaml",
                "imgsz": 640,
                "verbose": False,
            }
            return [result]

    annotated, people = yolo_pose.process_yolo_pose_frame(FakeModel(), frame)

    assert annotated.shape == frame.shape
    assert people[0]["track_id"] == 7
    assert people[0]["bbox"] == [1, 2, 30, 35]
    assert len(people[0]["keypoints"]) == 12
    assert people[0]["keypoints"][0] == {"x": 5, "y": 6, "confidence": 0.9}


def test_process_frame_uses_rtmpose_backend_interface():
    frame = object()
    expected = (object(), [{"track_id": 1}])

    class FakeRTMPosePipeline:
        def process_frame(self, received_frame, image_size):
            assert received_frame is frame
            assert image_size == yolo_pose.DEFAULT_IMAGE_SIZE
            return expected

    assert yolo_pose.process_frame(FakeRTMPosePipeline(), frame) == expected


@pytest.mark.parametrize(
    ("options", "expected_pipeline", "expected_model"),
    [
        (["--input", "video.mp4"], "rtmpose", "pose_pipeline.json"),
        (["--input", "video.mp4", "--pipeline", "yolo-pose"], "yolo-pose", "yolo11n-pose.pt"),
        (["--input", "video.mp4", "--model", "custom.pt"], "yolo-pose", "custom.pt"),
    ],
)
def test_fall_pipeline_selects_pipeline_and_default_model(
    options, expected_pipeline, expected_model, monkeypatch
):
    import fall_pipeline

    monkeypatch.setattr("sys.argv", ["fall_pipeline.py", *options])
    args = fall_pipeline.parse_args()

    assert args.pipeline == expected_pipeline
    assert Path(args.model).name == expected_model


@pytest.mark.parametrize("backend", ["rtmpose", "directml_rtmpose"])
def test_rtmpose_pipeline_runs_detector_tracker_then_pose(tmp_path, monkeypatch, backend):
    calls = []

    class FakeDetector:
        input_size = 640

        def __init__(self, model_path, device_id):
            pass

        def detect(self, frame):
            calls.append("detect")
            return np.array([[1, 2, 20, 30]]), np.array([0.95])

    class FakeTracker:
        def __init__(self):
            pass

        def update(self, boxes, scores, frame):
            calls.append("track")
            return np.array([[1, 2, 20, 30, 4, 0.95]])

    class FakePoseEstimator:
        def __init__(self, model_path, device_id):
            pass

        def estimate(self, frame, boxes):
            calls.append("pose")
            assert boxes == [[1, 2, 20, 30]]
            return [[{"x": 1.0, "y": 2.0, "confidence": 0.9}] * 12]

    pipeline_module = importlib.import_module("pose_estimators.rtmpose_pipeline")
    monkeypatch.setattr(pipeline_module, "YoloPersonDetector", FakeDetector)
    monkeypatch.setattr(pipeline_module, "PersonTracker", FakeTracker)
    monkeypatch.setattr(pipeline_module, "RTMPoseEstimator", FakePoseEstimator)

    config = tmp_path / "pipeline.json"
    config.write_text(
        json.dumps(
            {
                "backend": backend,
                "detector": "person.onnx",
                "pose": "pose.onnx",
            }
        ),
        encoding="utf-8",
    )
    pipeline = pipeline_module.RTMPosePipeline(config)
    annotated, people = pipeline.process_frame(np.zeros((32, 32, 3), dtype=np.uint8))

    assert calls == ["detect", "track", "pose"]
    assert annotated.shape == (32, 32, 3)
    assert people[0]["track_id"] == 4
    assert len(people[0]["keypoints"]) == 12
