"""Task 1.3 — visual EDA: sample grid per class and per-class histograms of width, height,
file size (plus aspect ratio, which informs the resize decision in Task 2.1).

Reads reports/audit/audit.csv; writes PNGs to reports/figures/.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.ticker as mticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

COLORS = {"ai": "#d1495b", "real": "#00798c"}


def sample_grid(df: pd.DataFrame, root: Path, out: Path, per_class: int, seed: int) -> None:
    classes = sorted(df["class"].unique())
    cols = per_class // 2
    fig, axes = plt.subplots(2 * len(classes), cols, figsize=(cols * 2.2, 2 * len(classes) * 2.3))
    for ci, cls in enumerate(classes):
        sample = df[df["class"] == cls].sample(per_class, random_state=seed)
        for k, row in enumerate(sample.itertuples()):
            ax = axes[2 * ci + k // cols, k % cols]
            with Image.open(root / row.path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((256, 256))
                ax.imshow(im)
            ax.set_title(f"{cls} · {row.width}×{row.height}", fontsize=8, color=COLORS[cls])
            ax.axis("off")
    fig.suptitle("Random samples per class (top: ai, bottom: real)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "eda_sample_grid.png", dpi=130)
    plt.close(fig)


def histograms(df: pd.DataFrame, out: Path) -> None:
    df = df.assign(file_size_kb=df.file_size / 1024, aspect_ratio=df.width / df.height)
    specs = [("width", "Width (px)", True), ("height", "Height (px)", True),
             ("file_size_kb", "File size (KB)", True), ("aspect_ratio", "Aspect ratio (w/h)", False)]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    for ax, (col, label, logx) in zip(axes, specs):
        values = df[col]
        bins = (np.logspace(np.log10(values.min()), np.log10(values.max()), 40) if logx
                else np.linspace(values.min(), min(values.max(), 3), 40))
        for cls, g in df.groupby("class"):
            ax.hist(g[col], bins=bins, alpha=0.55, label=f"{cls} (median {g[col].median():.0f})"
                    if col != "aspect_ratio" else f"{cls} (median {g[col].median():.2f})",
                    color=COLORS[cls])
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel(label + (" — log scale" if logx else ""))
        ax.set_ylabel("images")
        ax.legend(fontsize=8)
    fig.suptitle("Per-class distributions (readable images)")
    fig.tight_layout()
    fig.savefig(out / "eda_histograms.png", dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, default=Path("reports/audit/audit.csv"))
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("reports/figures"))
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.audit)
    df = df[df.readable]
    args.out.mkdir(parents=True, exist_ok=True)
    sample_grid(df, args.root, args.out, args.per_class, args.seed)
    histograms(df, args.out)
    print(f"wrote {args.out}/eda_sample_grid.png and {args.out}/eda_histograms.png")


if __name__ == "__main__":
    main()
