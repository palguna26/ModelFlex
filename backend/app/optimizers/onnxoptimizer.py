import time
import asyncio
from typing import Tuple, Dict
import onnx
import onnxruntime
from pathlib import Path
import numpy as np

async def optimize_onnx_model(
    model_path: str,
    target_device: str,
    output_path: str
) -> Tuple[str, Dict]:
    """Optimize ONNX model and measure performance."""

    # Validate model
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    original_size = Path(model_path).stat().st_size / (1024 * 1024)

    # Create inference session
    sess_options = onnxruntime.SessionOptions()
    original_session = onnxruntime.InferenceSession(
        model_path, sess_options, providers=['CPUExecutionProvider']
    )

    input_name = original_session.get_inputs()[0].name
    input_shape = [
        dim if isinstance(dim, int) and dim > 0 else 1
        for dim in original_session.get_inputs()[0].shape
    ]
    dummy_input = np.random.random(input_shape).astype(np.float32)

    # Measure original latency
    start_time = time.perf_counter()
    for _ in range(10):
        original_session.run(None, {input_name: dummy_input})
    original_latency = (time.perf_counter() - start_time) * 1000

    # Optimize
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = output_path

    available_providers = onnxruntime.get_available_providers()
    if target_device == "gpu" and "CUDAExecutionProvider" in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    _ = onnxruntime.InferenceSession(model_path, sess_options, providers=providers)

    optimized_size = Path(output_path).stat().st_size / (1024 * 1024)

    optimized_session = onnxruntime.InferenceSession(output_path, providers=['CPUExecutionProvider'])
    start_time = time.perf_counter()
    for _ in range(10):
        optimized_session.run(None, {input_name: dummy_input})
    optimized_latency = (time.perf_counter() - start_time) * 1000

    metrics = {
        "original_size_mb": round(original_size, 3),
        "optimized_size_mb": round(optimized_size, 3),
        "size_reduction_percent": round(((original_size - optimized_size) / original_size) * 100, 3),
        "original_latency_ms": round(original_latency, 3),
        "optimized_latency_ms": round(optimized_latency, 3),
        "latency_improvement_percent": round(((original_latency - optimized_latency) / original_latency) * 100, 3)
    }

    return output_path, metrics
