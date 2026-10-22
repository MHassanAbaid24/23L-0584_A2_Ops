"""Task 2.5 — threshold selection on validation data, metrics and figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from visionguard import CLASS_NAMES


def predict(model, ds) -> tuple[np.ndarray, np.ndarray]:
    """(y_true, p_ai) over a whole dataset, in dataset order."""
    ys, ps = [], []
    for x, y in ds:
        ys.append(np.asarray(y))
        ps.append(np.asarray(model(x, training=False)).reshape(-1))
    return np.concatenate(ys).astype(int), np.concatenate(ps)


def choose_threshold(y: np.ndarray, p: np.ndarray, beta: float = 0.5) -> dict:
    """Pick the decision threshold on VALIDATION data by maximising F-beta for the 'ai' class.

    beta < 1 weights precision of the "AI-generated" verdict above recall: calling a real photo fake
    (a false accusation) is treated as the worse error. Ties go to the higher threshold (more conservative).
    """
    candidates = np.unique(np.concatenate([[0.5], np.round(p, 4)]))
    best_t, best_s = 0.5, -1.0
    for t in candidates:
        s = fbeta_score(y, (p >= t).astype(int), beta=beta, zero_division=0)
        if s > best_s or (s == best_s and t > best_t):
            best_t, best_s = float(t), float(s)
    fpr, tpr, thr = roc_curve(y, p)
    youden_t = float(thr[np.argmax(tpr - fpr)])
    return {"threshold": best_t, "val_fbeta": best_s, "beta": beta, "youden_threshold": youden_t}


def metrics_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "real_called_fake_rate": fp / max(1, fp + tn),   # false positive rate on real images
        "fake_called_real_rate": fn / max(1, fn + tp),   # miss rate on AI images
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_confusion(y, p, threshold, title):
    cm = confusion_matrix(y, (p >= threshold).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], [f"pred {c}" for c in CLASS_NAMES])
    ax.set_yticks([0, 1], [f"true {c}" for c in CLASS_NAMES])
    ax.set_title(f"{title}\nthreshold={threshold:.3f}", fontsize=10)
    fig.tight_layout()
    return fig


def plot_roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], title):
    """curves: name → (y, p)."""
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate (real called AI)")
    ax.set_ylabel("True positive rate (AI caught)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def plot_history(history: dict[str, list[float]], best_epoch: int, title: str, stage_boundary: int | None = None):
    """Loss and accuracy curves with the best (lowest val_loss) epoch marked. Epochs are 1-based."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key in zip(axes, ("loss", "accuracy"), strict=True):
        ax.plot(epochs, history[key], label=f"train {key}")
        ax.plot(epochs, history[f"val_{key}"], label=f"val {key}")
        ax.axvline(best_epoch, color="red", ls="--", lw=1, label=f"best epoch {best_epoch}")
        if stage_boundary:
            ax.axvline(stage_boundary + 0.5, color="grey", ls=":", lw=1, label="fine-tuning starts")
        ax.set_xlabel("epoch")
        ax.set_ylabel(key)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig
