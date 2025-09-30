# app/model_optimizer.py

import tensorflow as tf
import torch
import onnx
import onnxruntime
import numpy as np
from pathlib import Path
import asyncio
from typing import Tuple, Dict

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
