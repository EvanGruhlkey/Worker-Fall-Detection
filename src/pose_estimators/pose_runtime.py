"""Choose a device and load ONNX models for detection and pose estimation."""

import logging

import onnxruntime as ort

logger = logging.getLogger(__name__)


def choose_provider():
    """Prefer NVIDIA CUDA, then DirectML, then CPU, based on installed support."""
    available = ort.get_available_providers()
    for provider in ("CUDAExecutionProvider", "DmlExecutionProvider"):
        if provider in available:
            return provider
    return "CPUExecutionProvider"


def load_session(model_path, device_id=0):
    """Load a model on the preferred device, allowing CPU fallback."""
    provider = choose_provider()
    options = ort.SessionOptions()
    options.add_free_dimension_override_by_name("batch", 1)
    if provider == "DmlExecutionProvider":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = ["CPUExecutionProvider"]
    if provider != "CPUExecutionProvider":
        providers.insert(0, (provider, {"device_id": device_id}))

    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=providers
    )
    logger.info("Loaded %s with %s", model_path, session.get_providers()[0])
    return session
