"""Task 2.1 figure: the same images before and after the train-time augmentation.

Uses the unshuffled val split purely for display, so both rows show the same images in the same order;
during training, augmentation is applied to the train split only.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visionguard import CLASS_NAMES
from visionguard.data import make_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--resize-mode", default="stretch", choices=["stretch", "pad"])
    ap.add_argument("--out", type=Path, default=Path("reports/figures/augmented_batch.png"))
    args = ap.parse_args()

    kw = dict(img_size=args.img_size, batch_size=1024, resize_mode=args.resize_mode, seed=7)
    # Same order, with and without augmentation: cache=False and no shuffle → compare like for like.
    plain = make_dataset("val", augment=False, cache=False, **kw)
    augmented = make_dataset("val", augment=True, cache=False, **kw)
    (x0, y), (x1, _) = next(iter(plain)), next(iter(augmented))
    # Half from each class, evenly spaced through the split.
    labels = np.asarray(y).astype(int)
    idx = np.concatenate([np.flatnonzero(labels == c)[:: max(1, (labels == c).sum() // (args.n // 2))]
                          [: args.n // 2] for c in (1, 0)])
    x0, x1, y = (np.take(np.asarray(a), idx, axis=0) for a in (x0, x1, y))

    fig, axes = plt.subplots(2, args.n, figsize=(args.n * 2.1, 4.8))
    for i in range(args.n):
        for row, x in enumerate((x0, x1)):
            ax = axes[row, i]
            ax.imshow(np.asarray(x[i]).astype("uint8"))
            ax.axis("off")
        axes[0, i].set_title(CLASS_NAMES[int(y[i])], fontsize=9)
    axes[0, 0].text(-0.15, 0.5, "resized", transform=axes[0, 0].transAxes, rotation=90, va="center")
    axes[1, 0].text(-0.15, 0.5, "augmented", transform=axes[1, 0].transAxes, rotation=90, va="center")
    fig.suptitle(f"One batch through the pipeline: {args.img_size}×{args.img_size}, resize={args.resize_mode} "
                 "(top) and after random flip/rotate/zoom/contrast (bottom)", fontsize=10)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}; batch shape {tuple(x1.shape)} dtype {x1.dtype}, "
          f"range [{float(np.min(x1)):.0f}, {float(np.max(x1)):.0f}]")


if __name__ == "__main__":
    main()
