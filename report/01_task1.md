# Chapter 1 — Data Understanding and Versioning (7 marks)

## 1.1 Environment

I made an isolated virtual environment, `.venv`, inside the project folder with Python 3.12.3. A project-local venv
keeps the exact library versions for this assignment separate from the system Python and from my other courses, and
anyone can rebuild it from the repository. `requirements.txt` pins every direct dependency to an exact version
(`tensorflow==2.20.0`, not `tensorflow>=2`). `requirements.lock.txt` is the full `pip freeze`, which also fixes the
transitive packages. I pinned versions because a minor update of Keras or MLflow can change default behaviour or the
format of a saved model, and then an old run could not be repeated.

To record the versions I wrote `scripts/env_report.py` and ran it with `.venv/bin/python scripts/env_report.py`. The
script prints the user, host, time and the interpreter path, so the screenshot shows it is my machine and this venv.
It reads each package version with `importlib.metadata.version`, which reports what is installed, not what a
requirements file asks for. It then asks TensorFlow for its physical GPU devices. Figure 1.1 shows the result:
TensorFlow 2.20.0, Keras 3.15.1, Keras Tuner 1.4.8, MLflow 3.17.2, DVC 3.67.1 and dagshub 0.7.3.

The GPU check prints `GPUs visible to TF : 0`. The laptop has a GTX 1050 Max-Q (3 GB), but no NVIDIA driver is loaded,
so CUDA fails with `CUDA_ERROR_NO_DEVICE` and TensorFlow falls back to the CPU. `TF built with CUDA : True` shows that the
TensorFlow build itself supports GPUs; only the driver is missing. Because of this I ran the heavy training on a Kaggle
GPU notebook and used the laptop for the audit, the EDA, the small toy labs and serving.

![Figure 1.1 — Exact library versions in the project venv and TensorFlow's GPU check (0 GPUs: no NVIDIA driver loaded, so TensorFlow runs on the CPU).](../reports/screenshots/task1/1.1_env_versions_gpu_check.png)

## 1.2 Dataset audit

I wrote `scripts/audit_dataset.py` to walk `data/raw/AiArtData` (label `ai`) and `data/raw/RealArt` (label `real`). It
writes one row per file to `reports/audit/audit.csv` and a summary to `reports/audit/summary.md`. Each row holds the
class, file size, width, height, colour mode, format and SHA-256. The summary is regenerated from the CSV, so the numbers
below come straight from the script.

**How it works.** Every file is opened twice with Pillow. First `verify()` checks the file structure without decoding.
`verify()` leaves the image object unusable, so the script opens the file again and calls `load()`, which decodes every
pixel. This second pass catches truncated files that pass `verify()`. Any exception marks the file as corrupt. The format
comes from the decoder, which reads the file's magic bytes, not from the file name. The SHA-256 is computed over the raw
bytes in 1 MB chunks. Two files with the same hash are byte-identical, so grouping by hash finds exact duplicates. The
script also flags hashes that appear in both classes, because such a pair would carry two contradictory labels.

**Table 1.1 — Files per class, readable and corrupt files, and files that belong to a duplicate group.**

| class | files | readable | corrupt | duplicate files |
|---|---|---|---|---|
| ai | 539 | 539 | 0 | 2 |
| real | 436 | 436 | 0 | 0 |
| TOTAL | 975 | 975 | 0 | 2 |

**Table 1.2 — Minimum, median and maximum of width, height (pixels) and file size (bytes) over the readable images.**

| class | metric | min | median | max |
|---|---|---|---|---|
| all | width | 183 | 1024 | 8495 |
| all | height | 135 | 768 | 7091 |
| all | file_size | 11311 | 169093 | 10195352 |
| ai | width | 225 | 1024 | 6000 |
| ai | height | 135 | 800 | 4001 |
| ai | file_size | 11738 | 199760 | 5626183 |
| real | width | 183 | 920 | 8495 |
| real | height | 158 | 720 | 7091 |
| real | file_size | 11311 | 144840 | 10195352 |

All 975 files decoded, so there are **0 corrupt files**. There is **one exact duplicate pair**:
`AiArtData/image33.jpg` and `AiArtData/image34.jpg` (163,866 bytes each). That is 1 redundant copy, and no hash appears
in both classes. The classes are moderately imbalanced (539 ai vs 436 real), so I report precision, recall and ROC-AUC
later and not only accuracy. Sizes vary widely, from 183 px to 8495 px wide, so every image must be resized to one
fixed input size.

**Format surprises.** The decoded formats are only JPEG (429 ai, 403 real) and PNG (110 ai, 33 real). However,
**20 files have an extension that does not match their content** (Table 1.3): `.webp` and `.gif` files that are
really JPEG, `.png` files that are JPEG and the reverse, and one JPEG with no extension at all. I took two decisions from
this. The input pipeline decodes by content (`tf.io.decode_image` reads the magic bytes in `src/visionguard/data.py`),
never by extension. The web app in Task 4 also validates uploads by content. Colour modes are mixed as well: besides RGB
there are palette (P: 14 ai, 9 real), RGBA (35 ai, 14 real) and greyscale (L: 1 ai, 6 real) images. So the decoder
forces three channels (`channels=3`) and the model always receives RGB. No file has more than one frame.

**Table 1.3 — The 20 files whose extension does not match the decoded format, grouped by type (full list in `reports/audit/summary.md`).**

| class | extension | real format | files |
|---|---|---|---|
| ai | png | JPEG | 8 |
| ai | jpg | PNG | 3 |
| ai | webp | JPEG | 3 |
| real | png | JPEG | 3 |
| real | jpg | PNG | 1 |
| real | gif | JPEG | 1 |
| real | (none) | JPEG | 1 |

**Cleaning and split.** `scripts/make_splits.py` reads `audit.csv` and applies three rules in order: drop corrupt
files, drop every copy of a hash that appears in both classes, and keep only the alphabetically first copy of any
other duplicate hash. Here only the third rule removed anything (one file), which leaves 974 images. The script then
makes a stratified 70/15/15 split with seed 42, using scikit-learn's `train_test_split` twice. The first call holds
out 30%, and the second splits that part in half. Stratifying keeps the ai/real ratio the same in every split, and the
fixed seed makes the lists reproducible. The split is done after removing duplicates, and the script asserts that no
SHA-256 appears in two splits. So no image can be in training and also in the test set. The lists are saved as
`splits/{train,val,test}.csv` (path, label, class, sha256), with label 1 = ai and 0 = real.

**Table 1.4 — Clean stratified split (seed 42), from `splits/cleaning_report.md`.**

| split | images | ai (1) | real (0) |
|---|---|---|---|
| train | 681 | 376 | 305 |
| val | 146 | 81 | 65 |
| test | 147 | 81 | 66 |

SHA-256 only finds byte-identical copies. A resized or re-compressed copy of the same picture gets a different hash,
so this audit cannot prove that there are no near-duplicates.

## 1.3 Visual EDA

`scripts/eda.py` reads `audit.csv` and draws two figures. The sample grid shows 12 random images per class (seed 42).
Before drawing, each image has its EXIF rotation applied, is converted to RGB and is shrunk to a thumbnail. The title
of each tile is the class and the original size. The histograms use a log scale for width, height and file size,
because these values cover almost two orders of magnitude. I added aspect ratio as a fourth panel because it decides
how images should be resized in Task 2.1.

Figure 1.2 shows the sample grid. Most AI-generated images are highly saturated digital art: psychedelic patterns,
painterly portraits and stylised animals with very smooth textures. The "real" class is more varied than its name
suggests. Besides photographs, it contains magazine covers, video thumbnails with letterboxing and photos of real
paintings. So the model must learn "made by a generator" vs "not made by a generator", not "art" vs "photo". Text
overlays and borders appear in both classes, so the model cannot use them as a cue. Some AI images are also composites
that include UI elements.

![Figure 1.2 — Twelve random images per class (top two rows: AI-generated, bottom two rows: real), seed 42; each title gives the class and the original size.](../reports/figures/eda_sample_grid.png)

Figure 1.3 shows the per-class distributions. AI images bunch at a few generator output sizes, which gives the width
histogram its tall spike near 1024 px. Real images are spread over a much wider range, from small web thumbnails to
8495-px camera originals. The real class has a cluster of tiny files (47 real images under 20 KB vs 15 AI images) and
fewer PNGs (8% vs 20%). Both classes have a median aspect ratio of 1.5, but square images are almost three times as
common in the AI class (22% vs 8%).

![Figure 1.3 — Per-class histograms of width, height and file size (log scale) and of aspect ratio; the legends give the per-class medians.](../reports/figures/eda_histograms.png)

**What this means for modelling.** These differences in size, format and shape are shortcuts. A model could score well
by learning "small, heavily compressed JPEG means real" instead of real generation artefacts. To reduce this risk,
every image is decoded from its bytes and resized to one fixed input size, and the model never sees file metadata.
The error analysis in Task 2.6 then checks whether the mistakes cluster on thumbnails or on unusual aspect ratios.

## 1.4 Data versioning with DVC

### Toy lab

I first learned DVC on a small CSV in a separate folder. The script `labs/dvc_toy_lab.sh` runs each step, prints the
command and a timestamp, and pauses so I could take a screenshot. It pushes to the DagsHub repository
`MHassanAbaid24/23L-0584_A2_Ops_dvc_toy`.

1. **Initialise.** In a new empty folder `dvc-toy` I ran `git init -b main` and `dvc init`, then committed DVC's
   skeleton (`.dvc/config`, `.dvc/.gitignore`, `.dvcignore`) as `ae58d01` (Figure 1.4). Git versions the code and the
   small pointer files. DVC versions the data, which lives in its own cache.
2. **Track v1.** I wrote a 3-row `toy.csv` and ran `dvc add toy.csv`. DVC moved the content into its cache, wrote the
   pointer file `toy.csv.dvc` and added `/toy.csv` to `.gitignore`, so Git cannot commit the data by mistake. I
   committed only the pointer and `.gitignore` (`9ffec16`, Figure 1.5).
3. **Change to v2.** I appended two rows, ran `dvc add` again and committed (`ee5c6fc`). Figure 1.6 shows the
   pointer diff: only the md5 and the size change, from 60 to 93 bytes.
4. **Go back to v1.** `git checkout HEAD~1 -- toy.csv.dvc` restores the old pointer, and `dvc checkout toy.csv`
   copies the matching bytes from the cache, so the 3-row file is back. The same two commands with `HEAD` returned to v2
   (Figure 1.7).
5. **Configure the remote.** I added DagsHub as the Git remote and as the default DVC remote
   (`dvc remote add -d origin …dvc_toy.dvc`). I stored the user name and token with `dvc remote modify --local`,
   which writes them to `.dvc/config.local`, a file that is git-ignored. Figure 1.8 shows that the committed
   `.dvc/config` holds only the URL.
6. **Push.** `git push` sent the commits, and `dvc push` uploaded the data of v2 and then of v1. `dvc status -c` reports
   that the cache and the remote are in sync (Figure 1.9).
7. **Delete and pull back.** I deleted `toy.csv` and the whole `.dvc/cache`, and `ls -la` shows that only the pointer
   was left. `dvc pull` downloaded v2 from DagsHub. After I checked out the v1 pointer, a second `dvc pull` brought
   v1 back as well (Figure 1.10). The data therefore survives losing the local copy, and both versions can be
   recovered from the remote.

Figure 1.11 shows the result on DagsHub: 4 commits, and `toy.csv` marked with a `dvc` badge. The file lives in DagsHub
storage, and Git holds only `toy.csv.dvc`.

![Figure 1.4 — Toy lab step 1: new empty folder `dvc-toy`, `git init`, `dvc init` and the first commit `ae58d01` (prompt and timestamp 2026-10-22 14:44:55 visible).](../reports/screenshots/task1/1.4_toy_01_init_git_dvc.png)

![Figure 1.5 — Step 2: v1 of `toy.csv` tracked with `dvc add`; the pointer file (md5 `00cd48…`, size 60) and the `/toy.csv` entry DVC added to `.gitignore`; commit `9ffec16`.](../reports/screenshots/task1/1.4_toy_02_track_v1_pointer_file.png)

![Figure 1.6 — Step 3: `git diff` of the pointer between v1 and v2; only the md5 and the size (60 → 93 bytes) change.](../reports/screenshots/task1/1.4_toy_03b_pointer_diff_v1_v2.png)

![Figure 1.7 — Step 4: returning to v1 (old pointer from Git, `dvc checkout` restores the 3-row file), then back to v2 (5 rows).](../reports/screenshots/task1/1.4_toy_04_return_to_v1_and_back.png)

![Figure 1.8 — Step 5: DagsHub Git and DVC remotes configured; the committed `.dvc/config` holds only the URL, while the credentials go to the git-ignored `.dvc/config.local`.](../reports/screenshots/task1/1.4_toy_05_configure_dagshub_remote.png)

![Figure 1.9 — Step 6: commits pushed to DagsHub Git; data of v2 and then v1 pushed to DagsHub storage; cache and remote in sync.](../reports/screenshots/task1/1.4_toy_06_push_both_versions.png)

![Figure 1.10 — Step 7: data file and the whole local cache deleted (`ls -la` shows only the pointer), then `dvc pull` restores v2 from DagsHub, and v1 after checking out the old pointer.](../reports/screenshots/task1/1.4_toy_07a_delete_data_cache_pull.png)

![Figure 1.11 — The toy repository on DagsHub: 4 commits, `toy.csv` carries a `dvc` badge (data in DagsHub storage, only `toy.csv.dvc` in Git).](../reports/screenshots/task1/1.4_toy_09_dagshub_repo.png)

### Applying it to the VisionGuard dataset

In the project I ran `dvc add data/raw`, committed the pointer `data/raw.dvc` and DVC's `data/.gitignore` to Git, and
ran `dvc push` to send the 504 MB of images to the project's DagsHub storage. The DagsHub project is connected to my
GitHub repository. Figure 1.12 shows the pointer, the `/raw` ignore rule, `git check-ignore` proving that an image and
`.dvc/config.local` are both ignored, the DagsHub remote, and the cache in sync with it. Figure 1.13 shows the `data/`
folder on DagsHub: `raw` carries the `dvc` badge, and Git holds only `raw.dvc` and `.gitignore`.

**What the pointer file holds.** `data/raw.dvc` contains:

- `md5: 8f23d6dc9b06d363c3981642c50856c0.dir`: the hash of the directory manifest, which lists every file's path and
  md5. The `.dir` suffix marks a whole folder rather than a single file.
- `size: 503783378`: the total size in bytes.
- `nfiles: 975`: the number of files.
- `hash: md5` and `path: raw`: the hash algorithm used and the tracked path, relative to the pointer.

These few lines are enough to fetch exactly this version from the cache or the remote. Checking out an old commit
and running `dvc pull` therefore gives back the dataset as it was at that commit. In Task 3.3 I log this md5 as the
data-version tag of each MLflow run.

**Why large data does not belong in Git.** Git stores every version of every file in full, in every clone, forever.
Changing a few images would make the repository larger for everyone. Diffs of binary images say nothing useful, and
GitHub rejects files over 100 MB. DVC keeps the Git repository small and still ties each commit to an exact data
version through the hash in the pointer.

![Figure 1.12 — Project evidence: Git history, the `data/raw.dvc` pointer (md5, 503,783,378 bytes, 975 files), DVC's `data/.gitignore`, proof that the images and `config.local` are git-ignored, the DagsHub remote, and cache ↔ remote in sync (user@host and timestamp 2026-10-22 16:12:12 visible).](../reports/screenshots/task1/1.4_apply_01_project_dvc.png)

![Figure 1.13 — DagsHub project `23L-0584_A2_Ops`, connected to GitHub: `data/raw` carries the `dvc` badge, while Git holds only `raw.dvc` and `.gitignore`.](../reports/screenshots/task1/1.4_apply_02_dagshub_data_folder.png)

### What went wrong

- **`dvc push --all-commits` silently skipped v1.** This option reads each commit's own `.dvc/config`. The v1 and v2
  commits were made before the remote was configured, so for those commits DVC had no remote to push to and skipped
  them without an error. I fixed it by checking out the v1 pointer and pushing it explicitly (`git checkout HEAD~2 --
  toy.csv.dvc`, then `dvc push`, visible in Figure 1.9). The pull in Figure 1.10 confirms that v1 really reached DagsHub.
- **The dataset pointer was hidden from DVC.** My first root `.gitignore` had the rule `/data/*`, which also ignored
  `data/raw.dvc`. DVC finds tracked data by scanning the pointers in the repository, so `dvc push` reported that no data
  was tracked. I removed the root rule and let `dvc add` write its own `data/.gitignore` containing only `/raw`
  (commit `c877ee4`). Now the images stay out of Git and the pointer is visible.
