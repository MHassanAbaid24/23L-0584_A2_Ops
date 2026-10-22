#!/usr/bin/env python3
"""Export a registered VisionGuard model alias for local serving."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import keras
import mlflow
import numpy as np
from mlflow import MlflowClient

from visionguard.tracking import setup_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", choices=("local", "dagshub"), default="dagshub")
    parser.add_argument("--name", default="23L-0584_A2_Ops_visionguard")
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    setup_tracking(args.tracking)
    version = MlflowClient().get_model_version_by_alias(args.name, args.alias)
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = Path(mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{version.run_id}/keras_model/model.keras", dst_path=str(args.out)
    ))
    champion_path = args.out / "champion.keras"
    model_path.replace(champion_path)
    config_path = Path(mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{version.run_id}/run_info/serving_config.json", dst_path=str(args.out)
    ))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        registered_model=args.name,
        version=version.version,
        alias=args.alias,
        run_id=version.run_id,
        exported_at=datetime.now(UTC).isoformat(),
    )
    (args.out / "serving_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    model = keras.models.load_model(champion_path)
    model.predict(np.zeros((1, config["img_size"], config["img_size"], 3), dtype=np.float32), verbose=0)
    with champion_path.open("rb") as model_file:
        digest = hashlib.file_digest(model_file, "sha256").hexdigest()
    print(f"model={args.name} version={version.version} alias={args.alias} run_id={version.run_id}")
    print(f"champion.keras sha256={digest} size={champion_path.stat().st_size}")


if __name__ == "__main__":
    main()
