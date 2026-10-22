"""Tune a small Fashion-MNIST classifier with Keras Tuner."""

from __future__ import annotations

import argparse
from pathlib import Path

import keras
import keras_tuner
import pandas as pd

TRAIN_SIZE = 12_000
VALIDATION_SIZE = 3_000
SEARCH_EPOCHS = 6
RETRAIN_EPOCHS = 3
SEED = 42
TUNER_DIRECTORY = "tuner"
PROJECT_NAME = "fashion_mnist_toy"
REPORT_DIRECTORY = Path("reports/tuner")


def parse_arguments() -> argparse.Namespace:
    """Parse tuner selection and trial-count options."""
    parser = argparse.ArgumentParser(
        description="Tune a toy Fashion-MNIST classifier with Keras Tuner."
    )
    parser.add_argument(
        "--algo",
        choices=("random", "hyperband"),
        default="hyperband",
        help="Search algorithm to use (default: hyperband).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Maximum RandomSearch trials (default: 10).",
    )
    return parser.parse_args()


def load_data() -> tuple[tuple[object, object], tuple[object, object]]:
    """Load and scale fixed Fashion-MNIST training and validation subsets."""
    (images, labels), _ = keras.datasets.fashion_mnist.load_data()
    images = images.astype("float32") / 255.0

    train_images = images[:TRAIN_SIZE]
    train_labels = labels[:TRAIN_SIZE]
    validation_start = TRAIN_SIZE
    validation_end = validation_start + VALIDATION_SIZE
    validation_images = images[validation_start:validation_end]
    validation_labels = labels[validation_start:validation_end]

    return (train_images, train_labels), (validation_images, validation_labels)


def build_model(hyperparameters: keras_tuner.HyperParameters) -> keras.Sequential:
    """Build the classifier whose dense width and learning rate are tuned."""
    units = hyperparameters.Int("units", min_value=32, max_value=256, step=32)
    learning_rate = hyperparameters.Choice(
        "learning_rate", values=[1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
    )

    model = keras.Sequential(
        [
            keras.layers.Flatten(),
            keras.layers.Dense(units, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_tuner(arguments: argparse.Namespace) -> keras_tuner.Tuner:
    """Create the requested tuner using the assigned on-disk configuration."""
    tuner_options = {
        "hypermodel": build_model,
        "objective": "val_accuracy",
        "directory": TUNER_DIRECTORY,
        "project_name": PROJECT_NAME,
        "overwrite": True,
    }
    if arguments.algo == "random":
        return keras_tuner.RandomSearch(max_trials=arguments.trials, **tuner_options)
    return keras_tuner.Hyperband(max_epochs=SEARCH_EPOCHS, factor=3, **tuner_options)


def trial_rows(tuner: keras_tuner.Tuner) -> list[dict[str, str | float]]:
    """Collect every trial's requested values in descending validation accuracy."""
    rows: list[dict[str, str | float]] = []
    for trial in tuner.oracle.trials.values():
        metric = trial.metrics.get_best_value("val_accuracy")
        if metric is None:
            continue
        values = trial.hyperparameters.values
        epochs = values.get("tuner/epochs", "")
        rows.append(
            {
                "trial_id": trial.trial_id,
                "units": str(values["units"]),
                "learning_rate": str(values["learning_rate"]),
                "epochs_trained": str(epochs),
                "best_val_accuracy": float(metric),
            }
        )
    return sorted(rows, key=lambda row: float(row["best_val_accuracy"]), reverse=True)


def format_markdown_table(rows: list[dict[str, str | float]]) -> str:
    """Render trial rows as a Markdown table without optional dependencies."""
    lines = [
        "| trial_id | units | learning_rate | epochs_trained | best_val_accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {trial_id} | {units} | {learning_rate} | {epochs_trained} | "
            "{best_val_accuracy:.6f} |".format(**row)
        )
    return "\n".join(lines)


def write_reports(
    algorithm: str,
    best_hyperparameters: keras_tuner.HyperParameters,
    rows: list[dict[str, str | float]],
    retrained_validation_accuracy: float,
) -> None:
    """Write the trial table and final retraining result to CSV and Markdown."""
    REPORT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    columns = [
        "trial_id",
        "units",
        "learning_rate",
        "epochs_trained",
        "best_val_accuracy",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        REPORT_DIRECTORY / "toy_results.csv", index=False
    )

    markdown = "\n".join(
        [
            "# Keras Tuner toy results",
            "",
            f"Algorithm used: `{algorithm}`",
            "",
            "## Best hyperparameters",
            "",
            f"- units: {best_hyperparameters.get('units')}",
            f"- learning_rate: {best_hyperparameters.get('learning_rate')}",
            "",
            "## All trials",
            "",
            format_markdown_table(rows),
            "",
            (
                "Retrained best model validation accuracy after "
                f"{RETRAIN_EPOCHS} epochs: {retrained_validation_accuracy:.6f}"
            ),
            "",
        ]
    )
    (REPORT_DIRECTORY / "toy_results.md").write_text(markdown, encoding="utf-8")


def main() -> None:
    """Run the search, summarize its trials, and retrain the best model."""
    arguments = parse_arguments()
    if arguments.trials < 1:
        raise ValueError("--trials must be at least 1")

    keras.utils.set_random_seed(SEED)
    (train_images, train_labels), validation_data = load_data()
    tuner = create_tuner(arguments)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=2, restore_best_weights=True
    )
    tuner.search(
        train_images,
        train_labels,
        validation_data=validation_data,
        epochs=SEARCH_EPOCHS,
        callbacks=[early_stopping],
        verbose=2,
    )
    tuner.results_summary(5)

    best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
    rows = trial_rows(tuner)
    best_model = tuner.hypermodel.build(best_hyperparameters)
    retraining_history = best_model.fit(
        train_images,
        train_labels,
        validation_data=validation_data,
        epochs=RETRAIN_EPOCHS,
        verbose=2,
    )
    retrained_validation_accuracy = retraining_history.history["val_accuracy"][-1]
    write_reports(
        arguments.algo,
        best_hyperparameters,
        rows,
        retrained_validation_accuracy,
    )
    print(
        "BEST: "
        f"units={best_hyperparameters.get('units')}, "
        f"learning_rate={best_hyperparameters.get('learning_rate')}"
    )


if __name__ == "__main__":
    main()
