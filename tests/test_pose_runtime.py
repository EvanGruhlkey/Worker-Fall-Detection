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
    monkeypatch.setattr(pose_runtime.ort, "get_available_providers", lambda: available)
    assert pose_runtime.choose_provider() == expected


def test_cpu_session_runs_model(monkeypatch):
    model_path = Path(__file__).parent / "fixtures" / "identity.onnx"
    monkeypatch.setattr(pose_runtime, "choose_provider", lambda: "CPUExecutionProvider")
    session = pose_runtime.load_session(model_path)
    result = session.run(None, {"input": np.array([[1, 2, 3]], dtype=np.float32)})
    np.testing.assert_array_equal(result[0], [[1, 2, 3]])
