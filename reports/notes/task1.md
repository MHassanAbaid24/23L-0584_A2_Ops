# Task 1 — draft notes for the report

These notes record the facts and numbers. Rewrite them in your own words for the PDF (the brief requires it).

## 1.1 Environment

- Isolated venv `.venv` (Python 3.12.3) inside the project; `requirements.txt` pins the direct dependencies,
  `requirements.lock.txt` is the full `pip freeze`.
- Versions: TensorFlow 2.20.0, Keras 3.15.1, Keras Tuner 1.4.8, MLflow 3.17.2, DVC 3.67.1, dagshub 0.7.3.
- GPU check: `GPUs visible to TF : 0`. The laptop has a GTX 1050 Max-Q (3 GB), but no NVIDIA driver is loaded
  (`cuInit: CUDA_ERROR_NO_DEVICE`), so TensorFlow runs on the CPU. Heavy training goes to a Kaggle GPU notebook.
- Command for the screenshot: `.venv/bin/python scripts/env_report.py`

## 1.2 Dataset audit (`scripts/audit_dataset.py` → `reports/audit/`)

- Each file is opened twice with Pillow: `verify()` checks the file structure, then a fresh `load()`
  decodes every pixel, which catches truncated files that pass `verify()`.
- Each file gets a SHA-256 of its raw bytes; equal hashes mean byte-identical duplicates.
- Counts: 975 files, 539 ai and 436 real; 0 corrupt.
- One exact duplicate pair: `AiArtData/image33.jpg` = `AiArtData/image34.jpg`. No hash appears in both classes.
- Width: min 183, median 1024, max 8495 px. Height: min 135, median 768, max 7091 px.
  File size: min 11 KB, median 165 KB, max 9.7 MB.
- Formats are JPEG and PNG only, but 20 files have an extension that does not match their content
  (for example, `.webp` and `.gif` files that are really JPEG, and a JPEG with no extension).
  So the pipeline must detect the format from the bytes, not from the file name.
- Colour modes: besides RGB there are P (palette), RGBA and L (greyscale), so every image is converted to
  3-channel RGB when it is decoded.
- Cleaning (`scripts/make_splits.py`) drops the redundant copy, leaving 974 images.
  Stratified 70/15/15 split with seed 42: train 681 (376 ai / 305 real), val 146 (81/65), test 147 (81/66).
  A check confirms that no hash appears in more than one split, so no image leaks between splits.

## 1.3 Visual EDA (`scripts/eda.py` → `reports/figures/`)

- `eda_sample_grid.png`: the AI class is dominated by saturated, painterly, psychedelic and portrait art.
  The real class mixes photographs with magazine covers, screenshots and paintings of real artworks,
  so "real" here means "not generated", not "photograph". Text overlays and borders appear in both classes,
  and some AI images are composites that contain UI elements.
- `eda_histograms.png` (log scale for width, height and size):
  - AI images cluster at generator output sizes; the spikes near 1024 px are typical of
    Stable Diffusion and Midjourney exports.
  - Real images are spread more widely and include a cluster of tiny thumbnails: 47 real files are under
    20 KB vs 15 AI files, most named `imagesNN.jpg`. 47 real images have a short side under 300 px vs 20 AI images.
  - Both classes have a median aspect ratio of 1.5. 22% of AI images are square vs 8% of real ones.
- Shortcut risk: PNG is 20% of AI files vs 8% of real files, and the classes differ in resolution and size.
  A model could learn "small, heavily compressed JPEG means real" instead of real generation artefacts.
  Resizing every image to one fixed size and training on decoded pixels (never metadata) limits this,
  and the error analysis in Task 2.6 checks for it.

## 1.4 DVC

- Toy lab: run `bash labs/dvc_toy_lab.sh` from the project root. It pauses after each step for a screenshot
  and uses the DagsHub repo `MHassanAbaid24/23L-0584_A2_Ops_dvc_toy`.
- What went wrong: `dvc push --all-commits` silently skipped v1. It reads each commit's own
  `.dvc/config`, and the v1/v2 commits were made before the remote existed. The fix was to check out the
  v1 pointer and push it explicitly.
- Apply: `dvc add data/raw` writes `data/raw.dvc`, which is committed to Git, while the 504 MB of images
  goes to DagsHub storage via `dvc push`.
  - What went wrong: a root `.gitignore` rule `/data/*` hid the pointer from DVC's repo scan, so
    `dvc push` reported "no data tracked". Letting DVC write its own `data/.gitignore` (containing `/raw`) fixed it.
- The pointer file holds:
  - the md5 of the directory listing (`8f23d6dc9b06d363c3981642c50856c0.dir`)
  - the total size (503,783,378 bytes)
  - the number of files (975)
  - the path

  That is enough to fetch exactly this version from the cache or remote.
- Why large data does not belong in Git:
  - Git stores every version of every file in full, inside every clone, forever.
  - Diffs on binary images are meaningless.
  - GitHub rejects files over 100 MB.

  DVC keeps Git small and still ties each commit to an exact data version, which is the md5 used as the
  data-version tag in Task 3.3.
- Credentials: the token is stored with `dvc remote modify --local`, which writes `.dvc/config.local`.
  That file is git-ignored, so only the remote URL is committed.
