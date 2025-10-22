import time
from pathlib import Path
from typing import Tuple, Dict
import tensorflow as tf
import numpy as np
import os

def get_model_size_mb(model_path: str) -> float:
    """Compute model size in MB, handling both file and directory models."""
    path = Path(model_path)
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total_size / (1024 * 1024)

def optimize_tensorflow_model(
    model_path: str,
    target_device: str,
    output_path: str
) -> Tuple[str, Dict]:
    """Optimize a TensorFlow model (Keras or SavedModel) with quantization or float16 compression."""

    model = tf.keras.models.load_model(model_path)
    original_size = get_model_size_mb(model_path)

    # Create dummy input
    input_shape = model.input_shape[1:]
    dummy_input = np.random.random((1,) + input_shape).astype(np.float32)

    # Measure original latency
    start_time = time.perf_counter()
    for _ in range(10):
        _ = model.predict(dummy_input)
    original_latency = (time.perf_counter() - start_time) * 1000  # ms

    # Representative dataset for int8 quantization
    def representative_dataset():
        for _ in range(100):
            sample = np.random.random((1,) + input_shape).astype(np.float32)
            yield [sample]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if target_device == "cpu":
        # Dynamic range INT8 quantization for CPU
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    elif target_device == "gpu":
        # Float16 quantization for GPU (Tensor Cores)
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
    else:
        # Default fallback (e.g., TPU)
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    # Try conversion
    try:
        optimized_model = converter.convert()
    except Exception as e:
        print("Quantization failed, falling back to float16:", e)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        optimized_model = converter.convert()

    # Save optimized model
    with open(output_path, "wb") as f:
        f.write(optimized_model)

    optimized_size = Path(output_path).stat().st_size / (1024 * 1024)

    # Measure optimized latency (TFLite)
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]
    dummy_input = np.random.random(input_shape).astype(input_dtype)

    start_time = time.perf_counter()
    for _ in range(10):
        interpreter.set_tensor(input_details[0]["index"], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
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
