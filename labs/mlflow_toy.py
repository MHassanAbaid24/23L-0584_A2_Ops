#!/usr/bin/env python3
"""Track tiny MNIST experiments locally or on DagsHub with MLflow."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
from dotenv import load_dotenv
from keras import backend, callbacks, datasets, layers, models, optimizers, utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_NAME = "toy-mnist"
BATCH_SIZE = 128
HIDDEN_UNITS = 64


class MlflowMetricsCallback(callbacks.Callback):
    """Write the four training and validation metrics at each epoch."""

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        if logs is None:
            return

        metrics = {
            metric_name: float(logs[metric_name])
            for metric_name in ("loss", "accuracy", "val_loss", "val_accuracy")
            if metric_name in logs
        }
        mlflow.log_metrics(metrics, step=epoch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=("local", "dagshub"),
        default="local",
        help="MLflow tracking destination (default: local)",
    )
    parser.add_argument(
        "--lrs",
        nargs="+",
        type=float,
        default=(1e-3, 1e-2),
        help="Learning rates; one MLflow run is created for each (default: 1e-3 1e-2)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs per run (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    return parser.parse_args()


def configure_tracking(target: str) -> str:
    """Configure the requested tracking store and select the lab experiment."""
    os.chdir(PROJECT_ROOT)

    if target == "local":
        tracking_uri = "sqlite:///mlflow.db"
    else:
        load_dotenv()
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise RuntimeError(
                "MLFLOW_TRACKING_URI is missing. Add it to the project-root .env file before "
                "using --target dagshub."
            )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return tracking_uri


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the small normalized MNIST split used by every run."""
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    return (
        train_images[:10_000].astype("float32") / 255.0,
        train_labels[:10_000],
        test_images[:2_000].astype("float32") / 255.0,
        test_labels[:2_000],
    )


def build_model(learning_rate: float) -> models.Sequential:
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(HIDDEN_UNITS, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def log_curves(history: callbacks.History) -> None:
    """Create and log a compact loss/accuracy training-curves artifact."""
    epoch_numbers = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epoch_numbers, history.history["loss"], label="train")
    axes[0].plot(epoch_numbers, history.history["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()

    axes[1].plot(epoch_numbers, history.history["accuracy"], label="train")
    axes[1].plot(epoch_numbers, history.history["val_accuracy"], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].legend()

    figure.tight_layout()
    mlflow.log_figure(figure, "curves.png")
    plt.close(figure)


def log_model(model: models.Sequential) -> None:
    """Log the Keras model, using the MLflow 3 API with legacy compatibility."""
    input_example = np.zeros((1, 28, 28), dtype=np.float32)
    try:
        mlflow.tensorflow.log_model(model, name="model", input_example=input_example)
    except TypeError as error:
        if "name" not in str(error):
            raise
        mlflow.tensorflow.log_model(model, artifact_path="model", input_example=input_example)


def dagshub_experiments_url(tracking_uri: str) -> str:
    """Map DagsHub's MLflow endpoint to its experiment-browser URL."""
    return f"{tracking_uri.rstrip('/')}/#/experiments"


def run_experiment(
    learning_rate: float,
    epochs: int,
    seed: int,
    target: str,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    validation_images: np.ndarray,
    validation_labels: np.ndarray,
) -> None:
    utils.set_random_seed(seed)
    model = build_model(learning_rate)

    with mlflow.start_run(run_name=f"lr-{learning_rate}") as run:
        mlflow.log_params(
            {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": BATCH_SIZE,
                "seed": seed,
                "hidden_units": HIDDEN_UNITS,
                "optimizer": "Adam",
            }
        )
        mlflow.set_tags({"lab": "3.1", "target": target, "author": "23L-0584"})
        history = model.fit(
            train_images,
            train_labels,
            validation_data=(validation_images, validation_labels),
            batch_size=BATCH_SIZE,
            epochs=epochs,
            callbacks=[MlflowMetricsCallback()],
            verbose=2,
        )
        log_curves(history)
        log_model(model)
        final_val_accuracy = history.history["val_accuracy"][-1]
        print(f"run_id={run.info.run_id} val_accuracy={final_val_accuracy:.6f}")

    backend.clear_session()


def main() -> None:
    args = parse_args()
    tracking_uri = configure_tracking(args.target)
    train_images, train_labels, validation_images, validation_labels = load_mnist()

    for learning_rate in args.lrs:
        run_experiment(
            learning_rate,
            args.epochs,
            args.seed,
            args.target,
            train_images,
            train_labels,
            validation_images,
            validation_labels,
        )

    if args.target == "local":
        print("Open the MLflow UI with: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    else:
        print(f"Open the DagsHub experiment browser: {dagshub_experiments_url(tracking_uri)}")


if __name__ == "__main__":
    main()
