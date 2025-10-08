import asyncio 
from pathlib import Path
from typing import Tuple, Dict
import tensorflow as tf
import numpy as np

# ---------------------- TensorFlow ----------------------


async def optimize_tensorflow_model(
    model_path: str,
    target_device: str,
    output_path: str
) -> Tuple[str, Dict]:
    """Optimize TensorFlow model"""
    model = tf.keras.models.load_model(model_path)

    # Get original metrics
    original_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB

    # Create dummy input
    input_shape = model.input_shape[1:]
    dummy_input = np.random.random((1,) + input_shape).astype(np.float32)

    # Measure original latency
    start_time = asyncio.get_event_loop().time()
    for _ in range(10):
        model.predict(dummy_input)
    original_latency = (asyncio.get_event_loop().time() - start_time) * 100  # ms

    # Representative dataset for int8 quantization
    def representative_dataset():
        for _ in range(100):
            sample_input = np.random.random((1,) + input_shape).astype(np.float32)
            yield [sample_input]

    # Apply optimizations
    if target_device == "cpu":
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        optimized_model = converter.convert()

    elif target_device == "gpu":
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        optimized_model = converter.convert()

    else:  # TPU / int8 fallback
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        try:
            optimized_model = converter.convert()
        except Exception:
            # fallback float16 if int8 fails
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            optimized_model = converter.convert()

    # Save optimized model
    with open(output_path, 'wb') as f:
        f.write(optimized_model)

    # Get optimized size
    optimized_size = Path(output_path).stat().st_size / (1024 * 1024)

    # Measure optimized latency
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']

    dummy_input = (np.random.random(input_shape) * 255).astype(input_dtype)

    start_time = asyncio.get_event_loop().time()
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
    optimized_latency = (asyncio.get_event_loop().time() - start_time) * 100  # ms

    metrics = {
        "original_size_mb": original_size,
        "optimized_size_mb": optimized_size,
        "size_reduction_percent": ((original_size - optimized_size) / original_size) * 100,
        "original_latency_ms": original_latency,
        "optimized_latency_ms": optimized_latency
    }

    return output_path, metrics