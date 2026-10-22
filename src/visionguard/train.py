"""Train, evaluate and (optionally) track one VisionGuard model.

    python -m visionguard.train --model cnn --tracking dagshub --logging manual
    python -m visionguard.train --model transfer --backbone efficientnetb0 --tracking dagshub --logging autolog

Covers Task 2.2–2.5 (models, training discipline, threshold + evaluation) and Task 3.3–3.4 (MLflow params,
per-epoch metrics, tags, test metrics, artifacts). Outputs go to outputs/<run_name>/ (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from visionguard.data import DEFAULT_DATA_ROOT, DEFAULT_SPLITS_DIR, make_dataset, read_split
from visionguard.evaluate import choose_threshold, metrics_at, plot_confusion, plot_history, plot_roc, predict
from visionguard.models import build_cnn, build_transfer, layer_table, unfreeze_top


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["cnn", "transfer"], default="cnn")
    ap.add_argument("--backbone", choices=["efficientnetb0", "mobilenetv2"], default="efficientnetb0")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="CNN lr, or head lr for transfer stage 1")
    ap.add_argument("--fine-tune-lr", type=float, default=1e-5, help="transfer stage 2 lr")
    ap.add_argument("--epochs", type=int, default=30, help="max epochs (stage 1 for transfer)")
    ap.add_argument("--fine-tune-epochs", type=int, default=15, help="max stage-2 epochs; 0 = no fine-tuning")
    ap.add_argument("--unfreeze-layers", type=int, default=None,
                    help="stage 2: unfreeze the last N backbone layers instead of the last block")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resize-mode", choices=["stretch", "pad"], default="stretch")
    ap.add_argument("--patience", type=int, default=6, help="early-stopping patience (epochs)")
    ap.add_argument("--threshold-beta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--test", action=argparse.BooleanOptionalAction, default=True,
                    help="evaluate on the test split (disable for tuning runs)")
    ap.add_argument("--tracking", choices=["none", "local", "dagshub"], default="none")
    ap.add_argument("--logging", choices=["manual", "autolog"], default="manual")
    ap.add_argument("--experiment", default="visionguard")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return ap.parse_args(argv)


def default_run_name(a: argparse.Namespace) -> str:
    core = "cnn" if a.model == "cnn" else a.backbone
    parts = [core, f"lr{a.lr:g}", f"img{a.img_size}", "aug" if a.augment else "noaug", a.resize_mode]
    if a.model == "transfer":
        parts.append(f"ft{a.fine_tune_lr:g}" if a.fine_tune_epochs else "frozen")
        if a.unfreeze_layers:
            parts.append(f"unf{a.unfreeze_layers}")
    return "-".join(parts) + "-" + a.logging + "-" + datetime.now().strftime("%m%d%H%M")


def compile_model(model: keras.Model, lr: float) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )


def model_summary_text(model: keras.Model) -> str:
    lines: list[str] = []
    model.summary(print_fn=lambda s, **_: lines.append(s), expand_nested=False, show_trainable=True)
    return "\n".join(lines)


def trainable_layer_names(model: keras.Model) -> list[str]:
    names = []
    for layer in model.layers:
        if isinstance(layer, keras.Model):
            names += [f"{layer.name}/{sub.name}" for sub in layer.layers if sub.trainable and sub.weights]
        elif layer.trainable and layer.weights:
            names.append(layer.name)
    return names


def run(a: argparse.Namespace) -> dict:
    keras.utils.set_random_seed(a.seed)
    if a.deterministic:
        tf.config.experimental.enable_op_determinism()
    a.run_name = a.run_name or default_run_name(a)
    out = a.out_dir / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    print(f"run {a.run_name} → {out}")

    data_kw = dict(img_size=a.img_size, batch_size=a.batch_size, resize_mode=a.resize_mode, seed=a.seed,
                   data_root=a.data_root, splits_dir=a.splits_dir)
    ds_train = make_dataset("train", augment=a.augment, **data_kw)
    ds_val = make_dataset("val", **data_kw)
    ds_test = make_dataset("test", **data_kw) if a.test else None

    # ---- model -------------------------------------------------------------------------------------
    base = None
    if a.model == "cnn":
        model = build_cnn(a.img_size, a.dropout)
    else:
        model, base = build_transfer(a.backbone, a.img_size, a.dropout)
    compile_model(model, a.lr)
    (out / "model_summary.txt").write_text(model_summary_text(model))
    pd.DataFrame(layer_table(model)).to_csv(out / "layer_table.csv", index=False)
    stage_layers = {"stage1_trainable": trainable_layer_names(model)}

    # ---- tracking ----------------------------------------------------------------------------------
    tracking = a.tracking != "none"
    if tracking:
        import mlflow

        from visionguard.tracking import MlflowEpochLogger, reproducibility_tags, setup_tracking

        setup_tracking(a.tracking, a.experiment)
        mlflow.start_run(run_name=a.run_name)
        params = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()
                  if k not in ("tracking", "out_dir", "run_name")}
        params.update(n_train=len(read_split("train", a.splits_dir)), n_val=len(read_split("val", a.splits_dir)),
                      n_test=len(read_split("test", a.splits_dir)), tf_version=tf.__version__,
                      keras_version=keras.__version__, optimizer="Adam", loss="binary_crossentropy",
                      total_params=model.count_params())
        if a.logging == "autolog":
            # Autolog records fit() params, per-epoch metrics, the model and its summary by itself;
            # the params block below is skipped so the run page shows what autolog captures on its own.
            mlflow.tensorflow.autolog(log_models=True, log_datasets=False)
        else:
            mlflow.log_params(params)
        mlflow.set_tags({**reproducibility_tags(a.seed), "logging_mode": a.logging,
                         "model_family": a.model, "author": "23L-0584"})

    ckpt_path = out / "best.keras"
    checkpoint = keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1)

    def callbacks():
        cbs = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=a.patience, restore_best_weights=True,
                                          verbose=1),
            checkpoint,  # one instance across both stages → keeps the global best val_loss
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=max(2, a.patience // 3),
                                              min_lr=1e-7, verbose=1),
        ]
        if tracking and a.logging == "manual":
            cbs.append(MlflowEpochLogger())
        return cbs

    # ---- training ----------------------------------------------------------------------------------
    h1 = model.fit(ds_train, validation_data=ds_val, epochs=a.epochs, callbacks=callbacks(), verbose=2)
    history = {k: list(map(float, v)) for k, v in h1.history.items()}
    stage_boundary = None
    if base is not None and a.fine_tune_epochs > 0:
        stage_boundary = len(history["loss"])
        unfrozen = unfreeze_top(base, a.backbone, a.unfreeze_layers)
        compile_model(model, a.fine_tune_lr)  # recompile so the new trainable flags take effect
        stage_layers["stage2_unfrozen_backbone_layers"] = unfrozen
        stage_layers["stage2_trainable"] = trainable_layer_names(model)
        (out / "model_summary_stage2.txt").write_text(model_summary_text(model))
        h2 = model.fit(ds_train, validation_data=ds_val, initial_epoch=stage_boundary,
                       epochs=stage_boundary + a.fine_tune_epochs, callbacks=callbacks(), verbose=2)
        for k, v in h2.history.items():
            history.setdefault(k, []).extend(map(float, v))
    (out / "trainable_layers.json").write_text(json.dumps(stage_layers, indent=2))
    (out / "history.json").write_text(json.dumps(history))

    best_epoch = int(min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__)) + 1
    model = keras.models.load_model(ckpt_path)  # global best by val_loss (across both stages)
    model.save(out / "model.keras")

    # ---- threshold on val, evaluate on test ---------------------------------------------------------
    y_val, p_val = predict(model, ds_val)
    thr = choose_threshold(y_val, p_val, a.threshold_beta)
    t = thr["threshold"]
    results = {"best_epoch": best_epoch, "epochs_run": len(history["loss"]), **thr,
               **{f"val_{k}": v for k, v in metrics_at(y_val, p_val, t).items()},
               **{f"val_at05_{k}": v for k, v in metrics_at(y_val, p_val, 0.5).items() if k in ("accuracy", "f1")}}
    figs = {"curves.png": plot_history(history, best_epoch, a.run_name, stage_boundary),
            "confusion_matrix_val.png": plot_confusion(y_val, p_val, t, "Validation")}
    roc_sets = {"val": (y_val, p_val)}
    if ds_test is not None:
        y_test, p_test = predict(model, ds_test)
        results.update({f"test_{k}": v for k, v in metrics_at(y_test, p_test, t).items()})
        figs["confusion_matrix_test.png"] = plot_confusion(y_test, p_test, t, "Test")
        roc_sets["test"] = (y_test, p_test)
        test_df = read_split("test", a.splits_dir).assign(p_ai=p_test, pred=(p_test >= t).astype(int))
        test_df.to_csv(out / "test_predictions.csv", index=False)
    figs["roc_curve.png"] = plot_roc(roc_sets, "ROC")
    for name, fig in figs.items():
        fig.savefig(out / name, dpi=120)
        plt.close(fig)

    serving = {"model_name": a.run_name, "threshold": t, "img_size": a.img_size, "resize_mode": a.resize_mode,
               "class_names": ["real", "ai"], "positive_class": "ai"}
    (out / "serving_config.json").write_text(json.dumps(serving, indent=2))
    (out / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    (out / "requirements.txt").write_text(
        subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout)

    # ---- log results + artifacts --------------------------------------------------------------------
    if tracking:
        numeric = {k: float(v) for k, v in results.items() if isinstance(v, int | float)}
        mlflow.log_metrics(numeric)
        mlflow.log_param("threshold", t) if a.logging == "manual" else mlflow.set_tag("threshold", f"{t:.4f}")
        for name in figs:
            mlflow.log_artifact(str(out / name), "figures")
        for split in ("train", "val", "test"):
            mlflow.log_artifact(str(a.splits_dir / f"{split}.csv"), "splits")
        for name in ("requirements.txt", "metrics.json", "serving_config.json", "model_summary.txt",
                     "layer_table.csv", "trainable_layers.json", "history.json"):
            mlflow.log_artifact(str(out / name), "run_info")
        mlflow.log_artifact(str(out / "model.keras"), "keras_model")
        if a.logging == "manual":
            mlflow.tensorflow.log_model(model, name="model")
        run_id = mlflow.active_run().info.run_id
        mlflow.end_run()
        results["mlflow_run_id"] = run_id
        if a.logging == "autolog":
            mlflow.tensorflow.autolog(disable=True)

    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in results.items()}, indent=1))
    return results


def main(argv=None) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    run(parse_args(argv))


if __name__ == "__main__":
    main()
