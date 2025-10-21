# app/model_optimizer.py

from pathlib import Path
from typing import Tuple, Dict
from backend.app.optimizers.tfoptimizer import optimize_tensorflow_model
from backend.app.optimizers.pytoptimizer import optimize_pytorch_model
from backend.app.optimizers.onnxoptimizer import optimize_onnx_model
async def optimize_model(
    model_path: str,
    target_device: str,
    output_path: str
) -> Tuple[str, Dict]:
    """
    Optimize an ML model for the target device.
    Returns the path to optimized model and performance metrics.
    """
    model_format = Path(model_path).suffix.lower()

    try:
        if model_format in ['.h5', '.pb']:
            return await optimize_tensorflow_model(model_path, target_device, output_path)
        elif model_format in ['.pt', '.pth']:
            return await optimize_pytorch_model(model_path, target_device, output_path)
        elif model_format == '.onnx':
            return await optimize_onnx_model(model_path, target_device, output_path)
        else:
            raise ValueError(f"Unsupported model format: {model_format}")

    except Exception as e:
        raise Exception(f"Model optimization failed: {str(e)}")

