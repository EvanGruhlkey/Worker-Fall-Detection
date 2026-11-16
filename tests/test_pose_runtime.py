from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")

from pose_estimators import pose_runtime


@pytest.mark.parametrize(("available", "expected"), [
    (["CPUExecutionProvider"], "CPUExecutionProvider"),
    (["DmlExecutionProvider", "CPUExecutionProvider"], "DmlExecutionProvider"),
    (["CPUExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider"], "CUDAExecutionProvider"),
    (["AzureExecutionProvider", "CPUExecutionProvider"], "CPUExecutionProvider"),
])
def test_choose_provider(available, expected, monkeypatch):
    monkeypatch.delenv("FALL_DEVICE", raising=False)
    monkeypatch.setattr(pose_runtime.ort, "get_available_providers", lambda: available)
    assert pose_runtime.choose_provider() == expected


def test_cpu_session_runs_model(monkeypatch):
    model_path = Path(__file__).parent / "fixtures" / "identity.onnx"
    monkeypatch.setattr(pose_runtime, "choose_provider", lambda: "CPUExecutionProvider")
    session = pose_runtime.load_session(model_path)
    result = session.run(None, {"input": np.array([[1, 2, 3]], dtype=np.float32)})
    np.testing.assert_array_equal(result[0], [[1, 2, 3]])


def test_directml_prefers_high_performance_gpu():
    # No device_id: ask Windows for the separate (fast) GPU, not GPU 0, which on
    # laptops with built-in graphics is usually the slower integrated one.
    options = pose_runtime.provider_options("DmlExecutionProvider")
    assert options == {"performance_preference": "high_performance", "device_filter": "gpu"}


def test_device_id_overrides_gpu_choice():
    assert pose_runtime.provider_options("DmlExecutionProvider", 1) == {"device_id": 1}
    assert pose_runtime.provider_options("CUDAExecutionProvider", 2) == {"device_id": 2}
    assert pose_runtime.provider_options("CUDAExecutionProvider") == {"device_id": 0}


def test_fall_device_cpu_forces_cpu(monkeypatch):
    monkeypatch.setattr(pose_runtime.ort, "get_available_providers",
                        lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setenv("FALL_DEVICE", "cpu")
    assert pose_runtime.choose_provider() == "CPUExecutionProvider"
    monkeypatch.setenv("FALL_DEVICE", "auto")
    assert pose_runtime.choose_provider() == "DmlExecutionProvider"
    monkeypatch.setenv("FALL_DEVICE", "")
    assert pose_runtime.choose_provider() == "DmlExecutionProvider"
