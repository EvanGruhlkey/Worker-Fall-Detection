"""Choose a device and load ONNX models for detection and pose estimation."""

import logging
import os

import onnxruntime as ort

logger = logging.getLogger(__name__)


def choose_provider():
    """Prefer NVIDIA CUDA, then DirectML, then CPU, based on installed support.

    ``FALL_DEVICE=cpu`` (environment or ``.env``) forces the CPU, e.g. to see how
    a laptop without a GPU performs. Empty or ``auto`` uses the order above.
    """
    device = os.environ.get("FALL_DEVICE", "auto").strip().lower() or "auto"
    if device == "cpu":
        return "CPUExecutionProvider"
    if device != "auto":
        logger.warning("FALL_DEVICE=%s not recognised (use auto or cpu); using auto", device)
    available = ort.get_available_providers()
    for provider in ("CUDAExecutionProvider", "DmlExecutionProvider"):
        if provider in available:
            return provider
    return "CPUExecutionProvider"


def provider_options(provider, device_id=None):
    """Which GPU to use. ``device_id`` (from the pipeline config) picks one by number.

    Without it, DirectML asks Windows for the high-performance GPU. On laptops
    with both built-in graphics and a separate GPU (e.g. Intel Iris Xe + NVIDIA),
    GPU 0 is usually the slower built-in one; the high-performance choice is the
    separate GPU, about twice as fast for the live pipeline.
    """
    if device_id is not None:
        return {"device_id": device_id}
    if provider == "DmlExecutionProvider":
        return {"performance_preference": "high_performance", "device_filter": "gpu"}
    return {"device_id": 0}


def load_session(model_path, device_id=None):
    """Load a model on the preferred device, allowing CPU fallback."""
    provider = choose_provider()
    options = ort.SessionOptions()
    options.add_free_dimension_override_by_name("batch", 1)
    if provider == "DmlExecutionProvider":
        options.enable_mem_pattern = False
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    providers = ["CPUExecutionProvider"]
    if provider != "CPUExecutionProvider":
        providers.insert(0, (provider, provider_options(provider, device_id)))

    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=providers
    )
    logger.info("Loaded %s with %s", model_path, session.get_providers()[0])
    return session
