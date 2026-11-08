import json

import numpy as np
import pytest
import torch

from sliding_window import TrackWindows
from temporal.features import extract_features, feature_names, window_signals
from temporal.model import FallGRU, load_checkpoint, save_checkpoint
from temporal.visualize import TrackSignals


def body_person():
    points = [{"x": 100.0, "y": 250.0, "confidence": 0.9} for _ in range(12)]
    points[0]["y"] = points[1]["y"] = 150.0
    return {"track_id": 1, "bbox": [50, 100, 150, 400], "confidence": 0.9, "keypoints": points}


def test_body_only_features_and_signals():
    windows = TrackWindows(window_size=3)
    for i in range(3):
        windows.update(i, [body_person()])
    window = windows.window(1)
    assert extract_features(window, 15).shape == (3, 91)
    assert feature_names()[-3:] == ["hip_height", "torso_angle", "hip_velocity"]
    np.testing.assert_allclose(window_signals(window, 15), [0.5, 0, 0])
    signals = TrackSignals(1)
    signals.append(0, 0, window_signals(window, 15), 1.0)
    assert not any("head" in key for key in signals.to_dict())


def test_old_feature_checkpoint_requires_retraining(tmp_path):
    path = tmp_path / "old.pt"
    save_checkpoint(path, FallGRU(hidden_size=8), fps=15, window_size=30)
    data = torch.load(path, weights_only=True)
    data["feature_version"] = 1
    torch.save(data, path)
    with pytest.raises(ValueError, match="retrain"):
        load_checkpoint(path)


@pytest.mark.parametrize("keypoint_format,count", [("COCO-17",17), ("COCO-body-12",12), (None,17)])
def test_pose_json_keeps_only_body_points(tmp_path, keypoint_format, count):
    from temporal.pose_io import load_pose_json
    from temporal.dataset import samples_from_pose_json
    from fall_pipeline import _json_frames

    person = body_person()
    if count == 17:
        person["keypoints"] = [{"x":9999,"y":9999,"confidence":0.9} for _ in range(5)] + person["keypoints"]
    metadata = {"fps":15}
    if keypoint_format:
        metadata["keypoint_format"] = keypoint_format
    path = tmp_path / "pose.json"
    path.write_text(json.dumps({"metadata":metadata, "frames":[{"frame_index":i,"people":[person]} for i in range(35)]}))
    data = load_pose_json(path)
    assert data.people[0][0]["keypoints"] == body_person()["keypoints"]
    samples, _ = samples_from_pose_json(path)
    assert samples and samples[0].features.shape == (30,91)
    assert len(next(_json_frames(path)[2])[1][0]["keypoints"]) == 12


def test_pose_json_rejects_unknown_keypoint_layout(tmp_path):
    from temporal.pose_io import load_pose_json
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps({"metadata":{"keypoint_format":"custom-12"},"frames":[]}))
    with pytest.raises(ValueError, match="keypoint format"):
        load_pose_json(path)


def test_committed_17_point_classifier_requires_retraining():
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "models" / "fall_gru.pt"
    with pytest.raises(ValueError, match="retrain"):
        load_checkpoint(path)


@pytest.mark.parametrize("keypoint_format,count", [("COCO-body-12",17), ("COCO-17",12), (None,11)])
def test_pose_json_rejects_mismatched_point_counts(tmp_path, keypoint_format, count):
    from temporal.pose_io import load_pose_json
    person = body_person()
    person["keypoints"] = [person["keypoints"][0]] * count
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"metadata":{"keypoint_format":keypoint_format}, "frames":[{"frame_index":0,"people":[person]}]}))
    with pytest.raises(ValueError, match="keypoints"):
        load_pose_json(path)
