"""Task 1.2 — dataset audit.

Walks data/raw/<class>/*, and for every file records class, file size, width, height,
colour mode, format and SHA-256. Flags unreadable files and exact duplicates.

Outputs (reports/audit/):
  audit.csv    one row per file
  summary.md   counts per class, min/median/max stats, corrupt + duplicate report
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd
from PIL import Image

CLASS_DIRS = {"AiArtData": "ai", "RealArt": "real"}  # folder name -> label
EXT_TO_FORMAT = {"jpg": "JPEG", "jpeg": "JPEG", "png": "PNG", "webp": "WEBP", "gif": "GIF"}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect(path: Path) -> dict:
    row = {"width": None, "height": None, "mode": None, "format": None, "n_frames": None,
           "readable": False, "error": ""}
    try:
        # Pass 1: verify() checks file structure without decoding pixels.
        with Image.open(path) as im:
            im.verify()
        # Pass 2: verify() leaves the object unusable, so reopen and fully decode the pixels.
        # This catches truncated files that pass verify().
        with Image.open(path) as im:
            im.load()
            row.update(width=im.width, height=im.height, mode=im.mode, format=im.format,
                       n_frames=getattr(im, "n_frames", 1), readable=True)
    except Exception as e:  # noqa: BLE001 — any decoder failure means "corrupt"
        row["error"] = f"{type(e).__name__}: {e}"
    return row


def audit(root: Path) -> pd.DataFrame:
    rows = []
    for folder, label in CLASS_DIRS.items():
        for path in sorted((root / folder).iterdir()):
            if not path.is_file():
                continue
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "class": label,
                "extension": path.suffix.lower().lstrip(".") or "(none)",
                "file_size": path.stat().st_size,
                "sha256": sha256_of(path),
                **inspect(path),
            })
    df = pd.DataFrame(rows)

    # Exact duplicates: same content hash. dup_group numbers each hash that occurs more than once.
    counts = df.groupby("sha256")["path"].transform("count")
    df["is_duplicate"] = counts > 1
    classes_per_hash = df.groupby("sha256")["class"].transform("nunique")
    # The extension is only a name; the decoder reports the real container format.
    df["extension_mismatch"] = df.readable & (df["extension"].map(EXT_TO_FORMAT) != df["format"])
    df["cross_class_duplicate"] = classes_per_hash > 1
    dup_hashes = {h: i for i, h in enumerate(sorted(df.loc[df.is_duplicate, "sha256"].unique()), 1)}
    df["dup_group"] = df["sha256"].map(dup_hashes).astype("Int64")
    return df


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)


def summarize(df: pd.DataFrame) -> str:
    ok = df[df.readable]
    out = ["# Dataset audit summary", ""]

    counts = df.groupby("class").agg(
        files=("path", "count"),
        readable=("readable", "sum"),
        corrupt=("readable", lambda s: int((~s).sum())),
        duplicate_files=("is_duplicate", "sum"),
    ).reset_index()
    counts.loc[len(counts)] = ["TOTAL", *counts.drop(columns="class").sum().tolist()]
    out += ["## Counts per class", "", md_table(counts), ""]

    stats = []
    for scope, part in [("all", ok), *[(c, g) for c, g in ok.groupby("class")]]:
        for col in ["width", "height", "file_size"]:
            stats.append([scope, col, int(part[col].min()), int(part[col].median()), int(part[col].max())])
    out += ["## Min / median / max (readable images)", "",
            md_table(pd.DataFrame(stats, columns=["class", "metric", "min", "median", "max"])), ""]

    out += ["## Formats and colour modes", "",
            md_table(ok.groupby(["class", "format"]).size().reset_index(name="files")), "",
            md_table(ok.groupby(["class", "mode"]).size().reset_index(name="files")), "",
            md_table(df.groupby(["class", "extension"]).size().reset_index(name="files")), ""]
    animated = ok[ok.n_frames > 1]
    out += [f"Animated / multi-frame files: {len(animated)}", ""]
    mism = df[df.extension_mismatch]
    out += [f"Files whose extension does not match the decoded format: {len(mism)} "
            "(so decode by content, never by extension)", ""]
    if len(mism):
        out += [md_table(mism[["path", "class", "extension", "format"]]), ""]

    corrupt = df[~df.readable]
    out += [f"## Corrupt / unreadable files: {len(corrupt)}", ""]
    if len(corrupt):
        out += [md_table(corrupt[["path", "class", "file_size", "error"]]), ""]

    dups = df[df.is_duplicate].sort_values(["dup_group", "path"])
    n_groups = dups["dup_group"].nunique()
    n_redundant = len(dups) - n_groups
    n_cross = dups.loc[dups.cross_class_duplicate, "dup_group"].nunique()
    out += [f"## Exact duplicates: {n_groups} groups, {len(dups)} files, "
            f"{n_redundant} redundant copies, {n_cross} groups spanning both classes", ""]
    if len(dups):
        out += [md_table(dups[["dup_group", "path", "class", "file_size"]]), ""]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out", type=Path, default=Path("reports/audit"))
    args = ap.parse_args()

    df = audit(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "audit.csv", index=False)
    summary = summarize(df)
    (args.out / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
