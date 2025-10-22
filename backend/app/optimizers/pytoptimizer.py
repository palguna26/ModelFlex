import torch
import time
from pathlib import Path
from typing import Tuple, Dict

def optimize_pytorch_model(
    model_path: str,
    target_device: str,
    output_path: str
) -> Tuple[str, Dict]:
    """Optimize PyTorch model using quantization or TorchScript tracing."""

    device = torch.device("cuda" if target_device == "gpu" and torch.cuda.is_available() else "cpu")

    # Load model safely
    model = torch.load(model_path, map_location=device)
    model.eval().to(device)

    original_size = Path(model_path).stat().st_size / (1024 * 1024)
    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(input_shape, device=device)

    # Measure original latency
    with torch.no_grad():
        start_time = time.perf_counter()
        for _ in range(10):
            _ = model(dummy_input)
        original_latency = (time.perf_counter() - start_time) * 1000  # ms

    # Optimization
    if device.type == "cpu":
        optimized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        scripted_model = torch.jit.script(optimized_model)
    else:
        with torch.no_grad():
            traced_model = torch.jit.trace(model, dummy_input)
            scripted_model = torch.jit.freeze(traced_model)

    torch.jit.save(scripted_model, output_path)

    optimized_size = Path(output_path).stat().st_size / (1024 * 1024)

    # Measure optimized latency
    with torch.no_grad():
        start_time = time.perf_counter()
        for _ in range(10):
            _ = scripted_model(dummy_input)
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
