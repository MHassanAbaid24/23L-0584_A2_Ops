"""Task 2.7 (apply) — Keras Tuner over the transfer model: fine-tune learning rate, dropout and number of
unfrozen backbone layers. Each trial = stage 1 (frozen backbone, head only) + stage 2 (fine-tune top N layers).

    python -m visionguard.tune --trials 10 --tracking dagshub

Objective: max val_auc (threshold-free, so trials are compared before any threshold is chosen).
The test split is never touched here. Results: reports/tuner/visionguard_results.{csv,md}.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import keras
import keras_tuner as kt
import pandas as pd
import tensorflow as tf

from visionguard.data import make_dataset
from visionguard.models import build_transfer, unfreeze_top
from visionguard.train import compile_model

UNFREEZE_CHOICES = [0, 10, 20, 40, 80]  # 0 = keep the backbone frozen (stage 2 trains the head only)
LR_CHOICES = [1e-4, 3e-5, 1e-5, 3e-6]
DROPOUT_CHOICES = [0.1, 0.3, 0.5]


class TwoStageTuner(kt.RandomSearch):
    """RandomSearch whose trial runs the same two-stage schedule as visionguard.train."""

    def __init__(self, *args, cfg: argparse.Namespace, ds_train, ds_val, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg, self.ds_train, self.ds_val = cfg, ds_train, ds_val

    def run_trial(self, trial, *args, **kwargs):
        hp, c = trial.hyperparameters, self.cfg
        keras.utils.set_random_seed(c.seed)
        model, base = build_transfer(c.backbone, c.img_size, hp.Choice("dropout", DROPOUT_CHOICES))
        early = lambda: [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,  # noqa: E731
                                                       restore_best_weights=True)]
        compile_model(model, c.head_lr)
        h1 = model.fit(self.ds_train, validation_data=self.ds_val, epochs=c.head_epochs, callbacks=early(),
                       verbose=0)
        n_unfreeze = hp.Choice("unfreeze_layers", UNFREEZE_CHOICES)
        if n_unfreeze:
            unfreeze_top(base, c.backbone, n_unfreeze)
        compile_model(model, hp.Choice("fine_tune_lr", LR_CHOICES))
        n1 = len(h1.history["loss"])
        h2 = model.fit(self.ds_train, validation_data=self.ds_val, initial_epoch=n1,
                       epochs=n1 + c.fine_tune_epochs, callbacks=early(), verbose=0)
        hist = {k: h1.history[k] + h2.history.get(k, []) for k in h1.history}
        best_i = min(range(len(hist["val_loss"])), key=hist["val_loss"].__getitem__)
        result = {"val_auc": float(hist["val_auc"][best_i]), "val_loss": float(hist["val_loss"][best_i]),
                  "val_accuracy": float(hist["val_accuracy"][best_i]), "best_epoch": best_i + 1,
                  "epochs_run": len(hist["loss"])}
        print(f"trial {trial.trial_id}: {hp.values} → {result}")
        if c.tracking != "none":
            import mlflow

            with mlflow.start_run(run_name=f"trial-{trial.trial_id}", nested=True):
                mlflow.log_params({**hp.values, "backbone": c.backbone, "img_size": c.img_size,
                                   "head_lr": c.head_lr})
                for step, v in enumerate(hist["val_auc"]):
                    mlflow.log_metric("epoch_val_auc", float(v), step=step)
                mlflow.log_metrics({k: float(v) for k, v in result.items()})
        return {"val_auc": result["val_auc"]}


def importance(df: pd.DataFrame, hps: list[str]) -> pd.DataFrame:
    """Crude sensitivity: range of the mean val_auc across each hyperparameter's values."""
    rows = []
    for hp in hps:
        means = df.groupby(hp)["val_auc"].mean()
        rows.append({"hyperparameter": hp, "values_tried": len(means), "best_value": means.idxmax(),
                     "mean_auc_spread": float(means.max() - means.min())})
    return pd.DataFrame(rows).sort_values("mean_auc_spread", ascending=False)


def main(argv=None) -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="efficientnetb0", choices=["efficientnetb0", "mobilenetv2"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--head-epochs", type=int, default=8)
    ap.add_argument("--fine-tune-epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tracking", choices=["none", "local", "dagshub"], default="none")
    ap.add_argument("--experiment", default="visionguard-tuning")
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("reports/tuner"))
    c = ap.parse_args(argv)

    keras.utils.set_random_seed(c.seed)
    kw = dict(img_size=c.img_size, batch_size=c.batch_size, seed=c.seed, data_root=c.data_root)
    ds_train, ds_val = make_dataset("train", **kw), make_dataset("val", **kw)

    tuner = TwoStageTuner(objective=kt.Objective("val_auc", "max"), max_trials=c.trials, seed=c.seed,
                          directory="tuner", project_name=f"visionguard_{c.backbone}", overwrite=True,
                          cfg=c, ds_train=ds_train, ds_val=ds_val)
    tuner.search_space_summary()

    parent = None
    if c.tracking != "none":
        import mlflow

        from visionguard.tracking import reproducibility_tags, setup_tracking

        setup_tracking(c.tracking, c.experiment)
        parent = mlflow.start_run(run_name=f"tuner-{c.backbone}-{c.trials}trials")
        mlflow.set_tags({**reproducibility_tags(c.seed), "author": "23L-0584", "lab": "2.7"})
        mlflow.log_params({k: str(v) for k, v in vars(c).items() if k not in ("tracking", "out")})

    tuner.search()

    rows = []
    for t in tuner.oracle.trials.values():
        if t.score is not None:
            rows.append({"trial": t.trial_id, **t.hyperparameters.values, "val_auc": t.score})
    df = pd.DataFrame(rows).sort_values("val_auc", ascending=False)
    imp = importance(df, ["fine_tune_lr", "dropout", "unfreeze_layers"])
    best = tuner.get_best_hyperparameters(1)[0].values

    c.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(c.out / "visionguard_results.csv", index=False)
    (c.out / "visionguard_results.md").write_text("\n".join([
        f"# VisionGuard tuner results ({c.backbone}, {len(df)} trials, RandomSearch, objective max val_auc)", "",
        df.to_markdown(index=False, floatfmt=".4f"), "", "## Sensitivity (spread of mean val_auc per value)", "",
        imp.to_markdown(index=False, floatfmt=".4f"), "", f"Best hyperparameters: `{json.dumps(best)}`", ""]))
    print(df.to_string(index=False))
    print(imp.to_string(index=False))
    print(f"BEST: {best}")

    if parent is not None:
        import mlflow

        mlflow.log_metric("best_val_auc", float(df["val_auc"].max()))
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_artifact(str(c.out / "visionguard_results.csv"), "tuner")
        mlflow.log_artifact(str(c.out / "visionguard_results.md"), "tuner")
        mlflow.end_run()
    tf.keras.backend.clear_session()


if __name__ == "__main__":
    main()
