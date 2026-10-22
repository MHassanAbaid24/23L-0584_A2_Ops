"""Task 2.6 error analysis + bonus Grad-CAM for one finished run (outputs/<run>/).

    python scripts/analyze_errors.py outputs/E0-base

Writes into the run dir:
  errors_grid.png     10 misclassified test images, annotated with the facts behind each hypothesis
  errors_table.csv    those images + audit metadata (size, format, mode, file size, aspect) for the write-up
  gradcam.png         Grad-CAM for 4 correct and 4 wrong test predictions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from visionguard.data import decode_and_resize


def load_rgb(path: Path, size: int, mode: str) -> np.ndarray:
    return decode_and_resize(tf.io.read_file(str(path)), size, mode).numpy()


def gradcam(model: keras.Model, img: np.ndarray) -> tuple[np.ndarray, float]:
    """Grad-CAM w.r.t. the last spatial feature map: the backbone output (transfer) or the last conv block (CNN).

    The models are linear chains, so we replay the layers one by one and watch the chosen feature map.
    """
    layers = model.layers[1:]  # skip InputLayer
    nested = [i for i, layer in enumerate(layers) if isinstance(layer, keras.Model)]
    if nested:
        target = nested[0]
    else:  # CNN: last layer before global pooling that still has spatial dims
        target = next(i for i, layer in enumerate(layers) if isinstance(layer, keras.layers.GlobalAveragePooling2D)) - 1
    x = tf.convert_to_tensor(img[None].astype("float32"))
    with tf.GradientTape() as tape:
        h = x
        for i, layer in enumerate(layers):
            h = layer(h, training=False)
            if i == target:
                fmap = h
                tape.watch(fmap)
        score = h[:, 0]
    grads = tape.gradient(score, fmap)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.nn.relu(tf.reduce_sum(fmap[0] * weights, axis=-1)).numpy()
    cam = cam / (cam.max() + 1e-8)
    cam = tf.image.resize(cam[..., None], img.shape[:2]).numpy()[..., 0]
    return cam, float(score[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--data-root", type=Path, default=Path("data/raw"))
    ap.add_argument("--audit", type=Path, default=Path("reports/audit/audit.csv"))
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    cfg = json.loads((a.run_dir / "serving_config.json").read_text())
    size, mode, thr = cfg["img_size"], cfg["resize_mode"], cfg["threshold"]
    preds = pd.read_csv(a.run_dir / "test_predictions.csv")
    audit = pd.read_csv(a.audit)[["path", "width", "height", "file_size", "format", "mode"]]
    preds = preds.merge(audit, on="path", how="left")
    preds["correct"] = preds["pred"] == preds["label"]
    preds["confidence_wrong"] = np.where(preds["label"] == 1, 1 - preds["p_ai"], preds["p_ai"])
    names = {0: "real", 1: "ai"}

    # ---- 10 misclassified, most confident mistakes first ------------------------------------------
    wrong = preds[~preds.correct].sort_values("confidence_wrong", ascending=False).head(a.n)
    wrong.assign(true=wrong.label.map(names), predicted=wrong.pred.map(names),
                 aspect=(wrong.width / wrong.height).round(2), file_kb=(wrong.file_size / 1024).round(1)) \
        .drop(columns=["correct", "confidence_wrong", "sha256"], errors="ignore") \
        .to_csv(a.run_dir / "errors_table.csv", index=False)
    cols = 5
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 4.0), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, r in zip(axes.ravel(), wrong.itertuples(), strict=False):
        ax.imshow(load_rgb(a.data_root / r.path, size, mode))
        ax.set_title(f"true {names[r.label]} → pred {names[r.pred]}  p(ai)={r.p_ai:.2f}\n"
                     f"{r.width}×{r.height} {r.format} {r.file_size / 1024:.0f}KB {r.mode}", fontsize=8)
    fig.suptitle(f"{len(wrong)} misclassified test images (threshold {thr:.3f}) — {cfg['model_name']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.run_dir / "errors_grid.png", dpi=120)
    plt.close(fig)

    # ---- Grad-CAM: 4 correct (2 per class) + 4 wrong ----------------------------------------------
    model = keras.models.load_model(a.run_dir / "model.keras")
    rng = np.random.default_rng(a.seed)
    correct = preds[preds.correct]
    pick_ok = pd.concat([g.sample(min(2, len(g)), random_state=int(rng.integers(1e6)))
                         for _, g in correct.groupby("label")])
    picks = pd.concat([pick_ok, wrong.head(4)])
    fig, axes = plt.subplots(2, len(picks), figsize=(len(picks) * 2.4, 5.2), squeeze=False)
    for j, r in enumerate(picks.itertuples()):
        img = load_rgb(a.data_root / r.path, size, mode)
        cam, p = gradcam(model, img.astype("float32"))
        axes[0, j].imshow(img)
        axes[1, j].imshow(img)
        axes[1, j].imshow(cam, cmap="jet", alpha=0.45)
        tag = "OK" if r.correct else "WRONG"
        axes[0, j].set_title(f"{tag}: true {names[r.label]}\np(ai)={p:.2f}", fontsize=8,
                             color="green" if r.correct else "red")
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle("Grad-CAM (bottom): where the evidence for 'AI-generated' comes from", fontsize=10)
    fig.tight_layout()
    fig.savefig(a.run_dir / "gradcam.png", dpi=120)
    plt.close(fig)
    print(f"wrote errors_grid.png, errors_table.csv, gradcam.png in {a.run_dir}")
    print(wrong[["path", "label", "p_ai", "width", "height", "format", "file_size"]].to_string(index=False))


if __name__ == "__main__":
    main()
