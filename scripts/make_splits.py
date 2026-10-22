"""Clean the audited dataset and write stratified train/val/test split lists.

Cleaning rules (applied to reports/audit/audit.csv):
  1. drop unreadable (corrupt) files;
  2. drop every copy of a hash that appears in BOTH classes (contradictory labels);
  3. keep one copy (first path alphabetically) of each remaining duplicate hash.

Split: stratified 70/15/15 with a fixed seed. Because each hash survives at most once,
no identical image can leak across splits (asserted below).

Outputs: splits/{train,val,test}.csv with columns path,label,class,sha256 (label 1 = ai, 0 = real),
and splits/cleaning_report.md.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = {"real": 0, "ai": 1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, default=Path("reports/audit/audit.csv"))
    ap.add_argument("--out", type=Path, default=Path("splits"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(args.audit)
    n0 = len(df)
    corrupt = df[~df.readable]
    df = df[df.readable]
    cross = df[df.cross_class_duplicate]
    df = df[~df.cross_class_duplicate]
    before_dedup = len(df)
    df = df.sort_values("path").drop_duplicates("sha256", keep="first")
    removed_dups = before_dedup - len(df)

    df = df.assign(label=df["class"].map(LABELS))[["path", "label", "class", "sha256"]]
    train, rest = train_test_split(df, test_size=args.val + args.test, stratify=df.label,
                                   random_state=args.seed)
    val, test = train_test_split(rest, test_size=args.test / (args.val + args.test),
                                 stratify=rest.label, random_state=args.seed)

    splits = {"train": train, "val": val, "test": test}
    hashes = [set(s.sha256) for s in splits.values()]
    assert not (hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2]), "leakage"

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, s in splits.items():
        s.sort_values("path").to_csv(args.out / f"{name}.csv", index=False)
        rows.append(f"| {name} | {len(s)} | {(s.label == 1).sum()} | {(s.label == 0).sum()} |")

    report = "\n".join([
        "# Cleaning and split report", "",
        f"- files audited: {n0}",
        f"- removed corrupt: {len(corrupt)}",
        f"- removed cross-class duplicate files: {len(cross)}",
        f"- removed redundant within-class copies: {removed_dups}",
        f"- clean images: {len(df)}",
        f"- seed: {args.seed}; stratified split {1 - args.val - args.test:.0%}/{args.val:.0%}/{args.test:.0%}",
        "", "| split | images | ai (1) | real (0) |", "|---|---|---|---|", *rows, "",
        "No SHA-256 hash appears in more than one split.",
    ])
    (args.out / "cleaning_report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
