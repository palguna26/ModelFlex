
import asyncio
from typing import Tuple, Dict
import onnx ,onnxruntime
from pathlib import Path
import numpy as np


# ---------------------- ONNX ----------------------

async def optimize_onnx_model(
    model_path: str,
    target_device: str,
    output_path: str
) -> Tuple[str, Dict]:
    """Optimize ONNX model"""
    model = onnx.load(model_path)
    onnx.checker.check_model(model)

    original_size = Path(model_path).stat().st_size / (1024 * 1024)

    sess_options = onnxruntime.SessionOptions()
    original_session = onnxruntime.InferenceSession(
        model_path, sess_options, providers=['CPUExecutionProvider']
    )

    input_name = original_session.get_inputs()[0].name
    input_shape = [dim if isinstance(dim, int) else 1 for dim in original_session.get_inputs()[0].shape]
    dummy_input = np.random.random(input_shape).astype(np.float32)

    start_time = asyncio.get_event_loop().time()
    for _ in range(10):
        original_session.run(None, {input_name: dummy_input})
    original_latency = (asyncio.get_event_loop().time() - start_time) * 100

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = output_path

    if target_device == "cpu":
        _ = onnxruntime.InferenceSession(
            model_path, sess_options, providers=['CPUExecutionProvider']
        )
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = [
            {
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'do_copy_in_default_stream': True,
            },
            {}
        ]
        _ = onnxruntime.InferenceSession(
            model_path, sess_options, providers=providers, provider_options=provider_options
        )

    optimized_size = Path(output_path).stat().st_size / (1024 * 1024)

    optimized_session = onnxruntime.InferenceSession(output_path, providers=['CPUExecutionProvider'])

    start_time = asyncio.get_event_loop().time()
    for _ in range(10):
        optimized_session.run(None, {input_name: dummy_input})
    optimized_latency = (asyncio.get_event_loop().time() - start_time) * 100

    metrics = {
        "original_size_mb": original_size,
        "optimized_size_mb": optimized_size,
        "size_reduction_percent": ((original_size - optimized_size) / original_size) * 100,
        "original_latency_ms": original_latency,
        "optimized_latency_ms": optimized_latency
    }

    return output_path, metrics
