#!/usr/bin/env python3
"""Predict one image with a registered VisionGuard model alias."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlflow
import tensorflow as tf
from mlflow import MlflowClient

from visionguard.data import decode_and_resize
from visionguard.tracking import setup_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--tracking", choices=("local", "dagshub"), default="dagshub")
    parser.add_argument("--name", default="23L-0584_A2_Ops_visionguard")
    parser.add_argument("--alias", default="champion")
    args = parser.parse_args()

    setup_tracking(args.tracking)
    version = MlflowClient().get_model_version_by_alias(args.name, args.alias)
    print(f"model={args.name} version={version.version} alias={args.alias} run_id={version.run_id}")
    config_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{version.run_id}/run_info/serving_config.json"
    )
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    load_started = time.perf_counter()
    model = mlflow.tensorflow.load_model(f"models:/{args.name}@{args.alias}")
    load_elapsed = time.perf_counter() - load_started
    image = decode_and_resize(tf.constant(args.image.read_bytes()), config["img_size"], config["resize_mode"])
    batch = tf.expand_dims(tf.cast(image, tf.float32), 0)
    predict_started = time.perf_counter()
    p_ai = float(model.predict(batch, verbose=0).reshape(-1)[0])
    predict_elapsed = time.perf_counter() - predict_started
    threshold = float(config["threshold"])
    label = "ai" if p_ai >= threshold else "real"
    confidence = p_ai if label == "ai" else 1 - p_ai
    print(f"image={args.image} p_ai={p_ai:.6f} threshold={threshold:.6f}")
    print(f"label={label} confidence={confidence:.6f}")
    print(f"load_seconds={load_elapsed:.3f} predict_seconds={predict_elapsed:.3f}")


if __name__ == "__main__":
    main()
