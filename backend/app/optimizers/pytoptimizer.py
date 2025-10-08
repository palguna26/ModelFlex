import torch
import onnx 
import onnxruntime
import numpy as np
from pathlib import Path
import asyncio
from typing import Tuple, Dict

# ---------------------- PyTorch ----------------------

async def optimize_pytorch_model(
    model_path: str,
    target_device: str,
    output_path: str
) -> Tuple[str, Dict]:
    """Optimize PyTorch model"""
    model = torch.load(model_path)
    model.eval()

    original_size = Path(model_path).stat().st_size / (1024 * 1024)

    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(input_shape)

    start_time = asyncio.get_event_loop().time()
    with torch.no_grad():
        for _ in range(10):
            model(dummy_input)
    original_latency = (asyncio.get_event_loop().time() - start_time) * 100

    if target_device == "cpu":
        model_quantized = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
        )
        model_quantized = torch.jit.script(model_quantized)
        torch.jit.save(model_quantized, output_path)
        optimized_model = model_quantized
    else:
        traced_model = torch.jit.trace(model, dummy_input)
        frozen_model = torch.jit.freeze(traced_model)
        torch.jit.save(frozen_model, output_path)
        optimized_model = frozen_model

    optimized_size = Path(output_path).stat().st_size / (1024 * 1024)

    start_time = asyncio.get_event_loop().time()
    with torch.no_grad():
        for _ in range(10):
            optimized_model(dummy_input)
    optimized_latency = (asyncio.get_event_loop().time() - start_time) * 100

    metrics = {
        "original_size_mb": original_size,
        "optimized_size_mb": optimized_size,
        "size_reduction_percent": ((original_size - optimized_size) / original_size) * 100,
        "original_latency_ms": original_latency,
        "optimized_latency_ms": optimized_latency
    }

    return output_path, metrics