#!/usr/bin/env python3
"""Register finished VisionGuard MLflow runs and assign deployment aliases."""

from __future__ import annotations

import argparse
import json
import time

import mlflow
from mlflow import MlflowClient

from visionguard.tracking import setup_tracking

DEFAULT_NAME = "23L-0584_A2_Ops_visionguard"
METRIC_TAGS = ("val_roc_auc", "val_f1", "test_roc_auc", "test_f1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking", choices=("local", "dagshub"), default="dagshub")
    parser.add_argument("--experiment", default="visionguard")
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--runs", nargs="+", required=True, metavar="RUN_NAME")
    parser.add_argument("--champion", required=True, metavar="RUN_NAME")
    parser.add_argument("--challenger", metavar="RUN_NAME")
    return parser.parse_args()


def finished_run(client: MlflowClient, experiment_id: str, run_name: str):
    escaped_name = run_name.replace("'", "\\'")
    matches = client.search_runs(
        [experiment_id],
        filter_string=f"attributes.status = 'FINISHED' AND tags.mlflow.runName = '{escaped_name}'",
    )
    if len(matches) != 1:
        raise SystemExit(f"Expected exactly one finished run named {run_name!r}; found {len(matches)}.")
    return matches[0]


def wait_until_ready(client: MlflowClient, name: str, version: str):
    for _ in range(60):
        model_version = client.get_model_version(name, version)
        if model_version.status == "READY":
            return model_version
        if model_version.status == "FAILED_REGISTRATION":
            raise RuntimeError(f"Registration of {name} version {version} failed.")
        time.sleep(1)
    raise TimeoutError(f"Timed out waiting for {name} version {version} to become READY.")


def version_for_run(client: MlflowClient, name: str, run_id: str):
    versions = client.search_model_versions(f"name='{name}'")
    matches = [version for version in versions if version.run_id == run_id]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple versions of {name!r} refer to run {run_id}.")
    return matches[0] if matches else None


def serving_config(run_id: str) -> dict[str, object]:
    path = mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/run_info/serving_config.json")
    with open(path, encoding="utf-8") as config_file:
        return json.load(config_file)


def register_run(client: MlflowClient, name: str, run):
    existing = version_for_run(client, name, run.info.run_id)
    model_version = existing or mlflow.register_model(f"runs:/{run.info.run_id}/model", name)
    model_version = wait_until_ready(client, name, model_version.version)
    config = serving_config(run.info.run_id)
    tags = {key: str(run.data.metrics[key]) for key in METRIC_TAGS if key in run.data.metrics}
    tags["run_name"] = run.data.tags["mlflow.runName"]
    tags["threshold"] = str(config["threshold"])
    for key, value in tags.items():
        client.set_model_version_tag(name, model_version.version, key, value)
    summary = (
        f"Run {tags['run_name']}: val ROC AUC={run.data.metrics.get('val_roc_auc', 'n/a')}, "
        f"test ROC AUC={run.data.metrics.get('test_roc_auc', 'n/a')}, threshold={config['threshold']}."
    )
    client.update_model_version(name, model_version.version, description=summary)
    return model_version, config


def print_table(rows: list[dict[str, str]]) -> None:
    columns = ("version", "run name", "run id", "aliases", "val_roc_auc", "test_roc_auc")
    widths = {column: max(len(column), *(len(row[column]) for row in rows)) for column in columns}
    print("  ".join(column.ljust(widths[column]) for column in columns))
    print("  ".join("-" * widths[column] for column in columns))
    for row in rows:
        print("  ".join(row[column].ljust(widths[column]) for column in columns))


def main() -> None:
    args = parse_args()
    requested_aliases = {"champion": args.champion}
    if args.challenger:
        requested_aliases["challenger"] = args.challenger
    unknown_alias_runs = set(requested_aliases.values()).difference(args.runs)
    if unknown_alias_runs:
        raise SystemExit(f"Alias targets must be in --runs: {', '.join(sorted(unknown_alias_runs))}")

    setup_tracking(args.tracking, args.experiment)
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(args.experiment)
    if experiment is None:
        raise SystemExit(f"Experiment {args.experiment!r} does not exist.")

    registered: dict[str, tuple[object, object]] = {}
    for run_name in args.runs:
        if run_name in registered:
            raise SystemExit(f"Duplicate run name in --runs: {run_name!r}")
        run = finished_run(client, experiment.experiment_id, run_name)
        registered[run_name] = (register_run(client, args.name, run)[0], run)

    for alias, run_name in requested_aliases.items():
        client.set_registered_model_alias(args.name, alias, registered[run_name][0].version)

    aliases_by_run = {run_name: [] for run_name in registered}
    for alias, run_name in requested_aliases.items():
        aliases_by_run[run_name].append(alias)
    rows = []
    for run_name, (model_version, run) in registered.items():
        rows.append({
            "version": str(model_version.version),
            "run name": run_name,
            "run id": run.info.run_id,
            "aliases": ",".join(aliases_by_run[run_name]) or "-",
            "val_roc_auc": str(run.data.metrics.get("val_roc_auc", "-")),
            "test_roc_auc": str(run.data.metrics.get("test_roc_auc", "-")),
        })
    print_table(rows)


if __name__ == "__main__":
    main()
