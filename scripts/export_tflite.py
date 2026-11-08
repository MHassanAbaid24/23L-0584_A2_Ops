#!/usr/bin/env python3
"""Export the VisionGuard champion as float32 TFLite and report test-set parity."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from visionguard.data import decode_and_resize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "cloud_api" / "model"
SUPPORTED_RESAMPLING = Image.Resampling.BILINEAR


def _interpreter(model_path: Path):
    """Use the deployment runtime when available, otherwise TensorFlow's interpreter."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        return tf.lite.Interpreter(model_path=str(model_path))
    return Interpreter(model_path=str(model_path))


def _pillow_input(image_path: Path, img_size: int) -> np.ndarray:
    """Decode and resize an image exactly as the cloud API does."""
    with Image.open(image_path) as image:
        image.seek(0)
        rgb = image.convert("RGB")
        if rgb.size != (img_size, img_size):
            rgb = rgb.resize((img_size, img_size), SUPPORTED_RESAMPLING)
        return np.asarray(rgb, dtype=np.float32)[np.newaxis, ...]


def main() -> None:
    config_path = MODEL_DIR / "serving_config.json"
    source_model = MODEL_DIR / "champion.keras"
    target_model = OUTPUT_DIR / "champion.tflite"
    target_config = OUTPUT_DIR / "serving_config.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(source_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    target_model.write_bytes(converter.convert())
    shutil.copy2(config_path, target_config)
    print(f"exported {target_model.relative_to(PROJECT_ROOT)} ({target_model.stat().st_size} bytes)")
    print(f"copied {target_config.relative_to(PROJECT_ROOT)}")

    interpreter = _interpreter(target_model)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    img_size = int(config["img_size"])
    threshold = float(config["threshold"])

    keras_probabilities: list[float] = []
    tflite_probabilities: list[float] = []
    labels: list[int] = []
    with (PROJECT_ROOT / "splits" / "test.csv").open(newline="", encoding="utf-8") as split_file:
        for row in csv.DictReader(split_file):
            image_path = PROJECT_ROOT / "data" / "raw" / row["path"]
            tf_input = tf.cast(
                decode_and_resize(tf.io.read_file(str(image_path)), img_size, str(config["resize_mode"])),
                tf.float32,
            )[tf.newaxis, ...]
            keras_probability = float(model(tf_input, training=False).numpy().reshape(-1)[0])

            interpreter.set_tensor(input_index, _pillow_input(image_path, img_size))
            interpreter.invoke()
            tflite_probability = float(interpreter.get_tensor(output_index).reshape(-1)[0])

            keras_probabilities.append(keras_probability)
            tflite_probabilities.append(tflite_probability)
            labels.append(int(row["label"]))

    keras_values = np.asarray(keras_probabilities)
    tflite_values = np.asarray(tflite_probabilities)
    truth = np.asarray(labels)
    absolute_delta = np.abs(keras_values - tflite_values)
    decision_changes = int(np.count_nonzero((keras_values >= threshold) != (tflite_values >= threshold)))
    keras_accuracy = float(np.mean((keras_values >= threshold) == truth))
    tflite_accuracy = float(np.mean((tflite_values >= threshold) == truth))
    print(f"parity images={truth.size} max|Δp|={absolute_delta.max():.7f} mean|Δp|={absolute_delta.mean():.7f}")
    print(f"decision_changes={decision_changes} threshold={threshold:.7f}")
    print(f"keras_accuracy={keras_accuracy:.4f} tflite_accuracy={tflite_accuracy:.4f}")


if __name__ == "__main__":
    main()
