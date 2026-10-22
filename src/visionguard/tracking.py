"""MLflow helpers: tracking target, reproducibility tags, per-epoch logging callback (Task 3.3)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import keras
import mlflow
import yaml

EXPERIMENT = "visionguard"


def setup_tracking(target: str, experiment: str = EXPERIMENT) -> str:
    """target: 'local' → sqlite file in the project; 'dagshub' → env vars (from the git-ignored .env)."""
    if target == "local":
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    elif target == "dagshub":
        from dotenv import load_dotenv

        load_dotenv()
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            raise SystemExit("MLFLOW_TRACKING_URI is not set (put it in .env or the environment)")
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    else:
        raise ValueError(target)
    mlflow.set_experiment(experiment)
    return mlflow.get_tracking_uri()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def reproducibility_tags(seed: int, dvc_file: Path = Path("data/raw.dvc")) -> dict[str, str]:
    """git commit (+dirty flag) and the DVC data version (md5 of the tracked dataset directory)."""
    tags = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": str(_git("status", "--porcelain", "--untracked-files=no") not in ("", "unknown")),
        "seed": str(seed),
    }
    try:
        out = yaml.safe_load(dvc_file.read_text())["outs"][0]
        tags.update(dvc_data_md5=out["md5"], dvc_data_nfiles=str(out.get("nfiles", "")))
    except (OSError, KeyError, TypeError):
        tags["dvc_data_md5"] = "unknown"
    return tags


class MlflowEpochLogger(keras.callbacks.Callback):
    """Manual per-epoch logging: every Keras log value + the current learning rate, step = global epoch."""

    def on_epoch_end(self, epoch, logs=None):
        metrics = {k: float(v) for k, v in (logs or {}).items()}
        metrics["lr"] = float(keras.ops.convert_to_numpy(self.model.optimizer.learning_rate))
        mlflow.log_metrics(metrics, step=epoch)
