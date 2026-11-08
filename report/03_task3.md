# Chapter 3 — Experiment Tracking with MLflow and DagsHub (12 marks)

## 3.1 MLflow locally

### What I did and why

Before I instrumented the real training code, I learned MLflow on a model small enough to train in seconds. `labs/mlflow_toy.py` trains a dense network on MNIST: Flatten → Dense(64, ReLU) → Dense(10, softmax), Adam, 5 epochs, batch 128. It uses the first 10,000 training images and the first 2,000 test images as the validation set. The script makes one MLflow run per learning rate (by default 1e-3 and 1e-2) in the experiment `toy-mnist`. Inside each run it:

- logs six parameters with `mlflow.log_params` (learning_rate, epochs, batch_size, seed, hidden_units, optimizer);
- logs loss, accuracy, val_loss and val_accuracy at the end of every epoch through a small Keras callback that calls `mlflow.log_metrics(..., step=epoch)`;
- sets the tags `lab`, `target` and `author`;
- saves a loss/accuracy plot with `mlflow.log_figure` as `curves.png` and the trained model with `mlflow.tensorflow.log_model`.

`python labs/mlflow_toy.py --target local` points MLflow at `sqlite:///mlflow.db`, a single database file in the project root. I chose a SQLite file over the older `mlruns/` folder store because one file is easy to inspect, back up and delete, and both `mlflow.db` and `mlruns/` are git-ignored. `mlflow ui --backend-store-uri sqlite:///mlflow.db` then starts the web UI on 127.0.0.1:5000, reading the same file.

Figure 3.1 shows the terminal output: lr 0.001 ended at val accuracy 0.8965 and lr 0.01 at 0.907. Figure 3.2 shows both runs in the local UI, each with a logged model. Figure 3.3 is the overview page of run `lr-0.01`, and Figure 3.4 is its `curves.png` artifact, where the validation loss rises after epoch 4. The larger learning rate fits faster and starts to overfit within five epochs. In the compare view (Figure 3.5) MLflow highlights `learning_rate` as the only parameter that differs. Figure 3.6 shows the parallel-coordinates plot of learning rate against final training loss.

![Figure 3.1 — `labs/mlflow_toy.py --target local`: two runs, lr 0.001 (val acc 0.8965) and lr 0.01 (0.907), with their run ids.](../reports/screenshots/task3/3.1_toy_two_local_runs_terminal.png)

![Figure 3.2 — Local MLflow UI, experiment `toy-mnist`: both runs, each with a logged model.](../reports/screenshots/task3/3.1_local_ui_runs_list.png)

![Figure 3.3 — Run `lr-0.01`: 6 parameters, 4 final metrics, tags `lab`, `target`, `author`; source `mlflow_toy.py` at commit d133853.](../reports/screenshots/task3/3.1_local_run_lr0.01_overview.png)

![Figure 3.4 — Artifact `curves.png` of run `lr-0.01`: train/val loss and accuracy per epoch; the validation loss rises after epoch 4 (overfitting at lr 0.01).](../reports/screenshots/task3/3.1_local_run_lr0.01_artifact_curves.png)

![Figure 3.5 — Compare view: only `learning_rate` differs (highlighted); lr 0.01 reaches val acc 0.907 vs 0.896 and val loss 0.332 vs 0.34.](../reports/screenshots/task3/3.1_local_compare_params_metrics.png)

![Figure 3.6 — Compare view: parallel-coordinates plot of learning rate vs final training loss.](../reports/screenshots/task3/3.1_local_compare_plot.png)

### Definitions in my own words

- **Experiment**: a named folder of runs that all try to answer one question, for example "which learning rate suits this MNIST net?" (`toy-mnist`) or "which VisionGuard model should ship?" (`visionguard`).
- **Run**: one execution of the training code, with everything recorded about it: what went in, what came out, when, and from which commit.
- **Parameter**: an input setting that is fixed before the run starts and does not change during it, such as the learning rate or the batch size. It is logged once.
- **Metric**: a number the run measures, such as loss or accuracy. It can be logged many times with a step number, which is how MLflow draws per-epoch curves.
- **Artifact**: any output file stored with the run, such as a plot, a CSV, a requirements list or the saved model.
- **Tag**: a free-form key–value label on the run. It is used for searching and for provenance, for example the git commit or who ran it, rather than as an input to training.

## 3.2 DagsHub setup

### What I did and why

The DagsHub project `23L-0584_A2_Ops` is connected to my GitHub repository, so every MLflow run that records a git commit links back to the code that produced it. DagsHub also hosts the DVC remote from Task 1.4, so code, data versions and experiments sit in one place. Each DagsHub repository exposes an MLflow tracking server at `https://dagshub.com/<user>/23L-0584_A2_Ops.mlflow`. The MLflow client finds it through three environment variables: `MLFLOW_TRACKING_URI`, plus `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` (my DagsHub user name and access token).

The credentials are never in code. They live in a `.env` file in the project root, which `python-dotenv` loads when a script is run with `--target dagshub` (`setup_tracking` in `src/visionguard/tracking.py`, `configure_tracking` in the toy). If the URI is missing, the scripts stop with a clear error instead of silently logging to a local folder. A committed `.env.example` lists the variable names with empty values, so the setup can be repeated without exposing the token. Figure 3.7 is the proof: `git check-ignore -v .env` shows that line 2 of `.gitignore` matches the file, `git status` is clean, and `.env.example` has 7 lines and no values. On Kaggle, where the real training ran, there is no `.env`. There, the notebook reads `DAGSHUB_USER` and `DAGSHUB_TOKEN` from Kaggle Secrets and puts them into the same environment variables, so the training code is the same in both places.

I then repeated the toy experiment with `python labs/mlflow_toy.py --target dagshub`, using 2 epochs to keep it quick: lr 1e-3 reached val accuracy 0.872 and lr 1e-2 reached 0.9105. Figure 3.8 shows the DagsHub Experiments tab with the toy runs `lr-0.001`/`lr-0.01` next to the VisionGuard runs, each linked to its commit. Figure 3.9 shows the same toy runs in DagsHub's hosted MLflow UI, with their logged models.

![Figure 3.7 — `git check-ignore -v .env` → matched by `.gitignore` line 2; `git status` is clean, so the credentials file has never been committed; `.env.example` (7 lines, no values) is the committed template.](../reports/screenshots/task3/3.2_env_gitignored_proof.png)

![Figure 3.8 — DagsHub project Experiments tab (22 runs), connected to GitHub: the toy runs `lr-0.001`/`lr-0.01` plus the VisionGuard runs, each linked to its Git commit.](../reports/screenshots/task3/3.2_dagshub_experiments_tab.png)

![Figure 3.9 — The same toy run on DagsHub's hosted MLflow (`…23L-0584_A2_Ops.mlflow`), experiment `toy-mnist`: both runs with logged models.](../reports/screenshots/task3/3.2_dagshub_mlflow_toy_runs.png)

## 3.3 Instrument training

### What I did and why

All VisionGuard training goes through `python -m visionguard.train`, which has two switches for tracking: `--tracking none|local|dagshub` (where to log) and `--logging manual|autolog` (how). To compare the two logging styles fairly, I trained the same configuration twice: E0-base with manual logging and E1-base-autolog with `mlflow.tensorflow.autolog`. Both use EfficientNetB0, 20 frozen epochs at lr 1e-3 and then 15 fine-tuning epochs at lr 1e-5.

Some things are logged in both modes, because the training code logs them itself:

- **Tags** from `reproducibility_tags()`: `git_commit` and `git_dirty` (whether tracked files had uncommitted changes), `seed` (42), and `dvc_data_md5`/`dvc_data_nfiles`, which are read from the `data/raw.dvc` pointer (md5 8f23d6dc…, 975 files). Together they say exactly which code and which data produced the run. A `logging_mode` tag tells the two styles apart.
- **Final metrics** after training: the validation threshold, validation and test accuracy, precision, recall, F1, ROC-AUC, confusion-matrix counts and the real-called-fake rate. The thresholded validation accuracy, precision and recall are now logged as `val_thr_*` (see "What went wrong" below).

**Manual (E0).** I log every command-line setting as a parameter, plus dataset sizes, TF/Keras versions and the parameter count: 30 parameters (Figure 3.10). They describe the experiment: backbone, image size, augmentation, resize mode, fine-tune lr, epochs per stage. A custom callback, `MlflowEpochLogger`, logs every Keras log value and the current optimizer learning rate at the end of each epoch. Its step is the global epoch index. Because the second `fit()` call starts at `initial_epoch=20`, the fine-tuning epochs continue at steps 20–34 instead of overwriting 0–14. Figure 3.11 shows the 39 metrics and the tags, and Figure 3.12 shows val_loss over all 35 epochs.

**Autolog (E1).** In this mode I skip my own `log_params` and callback, so the run page shows only what autolog collects by itself. The threshold is stored as a tag here, so that it does not mix into autolog's parameter list. Autolog also logged 31 parameters, but they describe `fit()` and the optimizer: `opt_name`, `opt_learning_rate`, `shuffle`, `validation_split`, `class_weight`, and `epochs=20, initial_epoch=0`, taken from the first `fit()` only (Figure 3.13). Nothing says which backbone, image size or augmentation was used. Figure 3.14 shows its biggest gap: val_loss stops at step 19. Autolog restarts its step counter on the second `fit()`, so the 15 fine-tuning epochs do not appear as epochs 20–34. Its `learning_rate` metric stays at the stage-1 value 0.001, while E0's callback logs the real final 1e-5 (Figure 3.15). Autolog did add things for free: a TensorBoard log, the model summary and checkpoints. I passed `log_datasets=False` because the DVC md5 tag already identifies the data more exactly than a digest of a tf.data pipeline.

**Conclusion.** Autolog is useful for zero-effort optimizer and `fit()` details. Manual logging is needed for everything that defines the experiment and for multi-stage training. Every other run therefore uses manual logging.

![Figure 3.10 — Manual logging (E0-base): parameters describe the experiment: model, backbone, img_size, lr, fine_tune_lr, epochs per stage, dropout, augment, resize_mode, patience, threshold_beta (30 in total).](../reports/screenshots/task3/3.3_E0_manual_params.png)

![Figure 3.11 — E0-base: 39 metrics (train/val/test), reproducibility tags git_commit, git_dirty=False, seed 42, dvc_data_md5, dvc_data_nfiles 975, logging_mode=manual; source train.py at 8b73f44; registered as model v1.](../reports/screenshots/task3/3.3_E0_manual_metrics_tags.png)

![Figure 3.12 — E0-base per-epoch curves: val_loss over all 35 epochs (both stages), val_precision per epoch; one-off values (val_fp) as bars.](../reports/screenshots/task3/3.3_E0_manual_metric_curves.png)

![Figure 3.13 — Autolog (E1-base-autolog): 31 parameters, but they are fit()/optimizer settings (batch_size None, validation_split, shuffle, class_weight, opt_name adam, opt_learning_rate 0.001); epochs=20 and initial_epoch=0 come from the first fit() call only. Nothing about the backbone, image size or augmentation.](../reports/screenshots/task3/3.3_E1_autolog_params.png)

![Figure 3.14 — E1-base-autolog curves: val_loss stops at step 19 (20 points); the 15 fine-tuning epochs are missing because autolog restarts the step counter on the second fit() call. Compare with the 35 points in E0.](../reports/screenshots/task3/3.3_E1_autolog_metric_curves.png)

![Figure 3.15 — E1-base-autolog: the same reproducibility tags (git_commit, git_dirty, seed, dvc_data_md5, dvc_data_nfiles, logging_mode=autolog, threshold), which come from the training code, not from autolog. Autolog's `learning_rate` metric is stale at 0.001 (the stage-1 value). Not registered.](../reports/screenshots/task3/3.3_E1_autolog_metrics_tags.png)

### What went wrong

The training code first logged the final, thresholded validation metrics as `val_accuracy`, `val_precision` and `val_recall`. Keras logs metrics with exactly these names once per epoch, at threshold 0.5, through my callback or through autolog. When a metric has several values, MLflow shows the one with the highest step. So the run pages showed the last epoch's Keras value, not the thresholded one. For example, E5-mobilenetv2 showed val_precision 0.8148 on DagsHub, while the thresholded value in its `metrics.json` is 0.9032. For E0-base, 0.7816 was shown against a real value of 0.8451.

Metrics with unique names were not affected: `val_roc_auc`, `val_f1`, `val_real_called_fake_rate`, `val_fake_called_real_rate` and every `test_*` metric. The model choice in 3.5, which rests on validation ROC-AUC and the real-called-fake rate, is therefore unchanged. Every other column of `reports/experiments/comparison.csv` was re-checked against each run's `metrics.json`, and all of them match.

As a fix, `src/visionguard/train.py` now logs these three as `val_thr_accuracy`, `val_thr_precision` and `val_thr_recall`, while `metrics.json` keeps the plain names. I also rebuilt the `val_precision` column of `reports/experiments/comparison.md|csv` from `metrics.json`; it now reads, for example, E5 0.9032, E7 0.8354 and E0 0.8451. The existing DagsHub runs were not re-run, so their run pages still show the last-epoch values under `val_precision`. The lesson: metric names are an interface, and a summary metric must never reuse the name of a per-step metric.

## 3.4 Artifacts

At the end of every run, `src/visionguard/train.py` uploads the run's files to MLflow with `mlflow.log_artifact`, grouped into folders:

- `figures/`: training curves, validation and test confusion matrices, ROC curve;
- `splits/`: the train/val/test CSV lists, so the exact images behind every metric can be traced;
- `run_info/`: `requirements.txt` (a `pip freeze` of the training environment), `metrics.json`, `serving_config.json` (threshold, image size, resize mode, class names), model summary, layer table, trainable-layer lists and the epoch history;
- `keras_model/model.keras`: the plain Keras file of the best checkpoint;
- `model`: the MLflow-format model, which is the one the registry points to.

I log the model twice on purpose. The MLflow model is what the registry and `mlflow.tensorflow.load_model` work with. The single `.keras` file is what the web server loads without needing MLflow (see 3.6). The serving config sits next to it, so the threshold chosen on validation always travels with the weights it belongs to.

Figure 3.16 shows the artifact tree of the champion run E5-mobilenetv2 in DagsHub's artifact store. Figure 3.17 opens its test confusion matrix: at threshold 0.673 it has 60 TN, 6 FP (real called AI), 26 FN and 55 TP. Figure 3.18 shows `requirements.txt` (18 KB). It includes Kaggle's preinstalled packages, so it records the training environment exactly, but it is not a slim list for serving.

![Figure 3.16 — Champion run E5-mobilenetv2, Artifacts: `figures/` (val/test confusion matrices, ROC, curves), `keras_model/model.keras`, `run_info/` (requirements.txt, metrics, serving_config, model summary, layer table, trainable layers, history), `splits/` (train/val/test CSVs), stored in DagsHub's MLflow artifact store.](../reports/screenshots/task3/3.4_E5_artifacts_tree.png)

![Figure 3.17 — Test confusion matrix of E5-mobilenetv2 at threshold 0.673: 60 TN, 6 FP (real called AI), 26 FN, 55 TP.](../reports/screenshots/task3/3.4_E5_artifact_confusion_matrix_test.png)

![Figure 3.18 — `run_info/requirements.txt`: the full `pip freeze` of the Kaggle training environment (18 KB), so the run can be reproduced exactly.](../reports/screenshots/task3/3.4_E5_artifact_requirements.png)

## 3.5 Experiment comparison

### What I did

The Kaggle notebook (`notebooks/kaggle_train.ipynb`) runs an experiment matrix. E0 is the base configuration: EfficientNetB0, augmentation, stretch resize, 224 px, fine-tune lr 1e-5, manual logging. Each of E1–E7 adds exactly one flag to it, so any difference in the metrics can be traced to that one change. The from-scratch `cnn-baseline` from Task 2.2 is logged to the same experiment as a reference. The notebook then pulls all runs with `mlflow.search_runs` and writes `reports/experiments/comparison.md|csv`. Table 3.1 is an extract of it, with values copied exactly.

**Table 3.1 — VisionGuard runs (experiment `visionguard`). "val/test r→f" is the rate of real images called AI at the run's validation-chosen threshold. Sorted by val AUC.**

| run | change vs E0 | val AUC | val F1 | val r→f | test AUC | test F1 | test r→f |
|---|---|---|---|---|---|---|---|
| E5-mobilenetv2 | MobileNetV2 backbone | **0.8807** | 0.7832 | **0.0923** | 0.8709 | 0.7746 | 0.0909 |
| E7-ft-lr-1e-4 | fine-tune lr 1e-4 | 0.8786 | 0.8250 | 0.2000 | 0.8902 | 0.8364 | 0.2273 |
| E2-no-augment | no augmentation | 0.8764 | 0.8471 | 0.2615 | 0.8915 | 0.8362 | 0.3333 |
| E0-base | — | 0.8697 | 0.7895 | 0.1692 | 0.8853 | 0.8153 | 0.1818 |
| E1-base-autolog | logging only (autolog) | 0.8663 | 0.7922 | 0.1846 | 0.8844 | 0.8250 | 0.1970 |
| E6-resize-pad | pad instead of stretch | 0.8587 | 0.7975 | 0.2154 | 0.8883 | 0.8503 | 0.2273 |
| E3-frozen-only | no fine-tuning | 0.8572 | 0.8324 | 0.3077 | 0.8739 | 0.8046 | 0.3485 |
| E4-img160 | 160 px input | 0.8298 | 0.7826 | 0.2615 | 0.8694 | 0.8199 | 0.2121 |
| cnn-baseline | from-scratch CNN (reference) | 0.6957 | 0.6711 | 0.3077 | 0.7421 | 0.6933 | 0.2576 |

Figure 3.19 shows the same ranking in the DagsHub runs table. Figure 3.20 plots validation ROC-AUC against the real-called-fake rate. Figure 3.21 is the parallel-coordinates view, where the low line runs through img_size 160 (E4).

What the single changes show: fine-tuning the last block helps (E3 < E0); smaller images hurt (E4); padding instead of stretching makes little difference (E6); a 10× larger fine-tune lr helps (E7); and augmentation did not help on this small validation set (E2). E1 differs only in logging, and its numbers stay within about 0.004 AUC of E0 (0.8663 vs 0.8697). This suggests that differences of that size are noise.

### Choice and justification

I chose **E5-mobilenetv2** as champion and kept E7 as challenger. I used two metrics, both measured on validation:

1. **Validation ROC-AUC**, because it measures ranking quality independently of the threshold. E5 is highest at 0.8807.
2. **Real-called-fake rate** at the threshold chosen on validation, because in Task 2.5 I decided that falsely accusing a real photo is the worse error. E5's 0.0923 is about half of the next-best rate (E0, 0.1692). In Figure 3.20, E5 is the point alone at the bottom right.

E5 is also the smallest model (16.7 MB `.keras`, 2.3 M parameters), which suits a CPU server. The trade-off is that it misses more AI images (val fake→real 0.3086). F1 alone would favour E2 or E3, but those models call 26–31 % of real validation images fake.

Selection used validation only. The test columns are reported, not used for selection. On test, E5's AUC (0.8709) is below several EfficientNet runs. With 147 test images these gaps of about 0.02 are within noise, and picking the run with the best test score would leak the test set into model selection. Its low false-accusation rate did hold on test (0.0909).

![Figure 3.19 — Runs table sorted by validation ROC-AUC, with test F1/AUC, val F1, val real→fake rate, backbone and img size: E5-mobilenetv2 leads on val AUC (0.881) with the lowest real→fake rate (0.092); cnn-baseline is last (0.696). The CNN's `backbone` cell is the unused default.](../reports/screenshots/task3/3.5_runs_table_sorted_by_val_auc.png)

![Figure 3.20 — Compare view of E0–E7: validation ROC-AUC (x) vs rate of real images called AI (y). The point at (0.881, 0.092), bottom-right and alone, is E5-mobilenetv2: best ranking quality and fewest false accusations.](../reports/screenshots/task3/3.5_compare_scatter_auc_vs_real_called_fake.png)

![Figure 3.21 — Parallel coordinates for E0–E7: backbone, img_size, augment, resize_mode, fine_tune_lr, fine_tune_epochs → val ROC-AUC (colour). The dark-blue low line is img_size 160 (E4); the line marked "unknown" is E1, because autolog logs no experiment parameters.](../reports/screenshots/task3/3.5_compare_parallel_coordinates.png)

### What went wrong

- The autolog run E1 has no `backbone`, `img_size` or `augment` parameters, so it shows up as "unknown" in Figure 3.21. To fill these columns in the comparison table, the notebook works out E1's settings from its run name instead of reading them from MLflow. This is another cost of autolog.
- Running the notebook a second time created duplicate runs with the same names on DagsHub. Duplicates make the comparison ambiguous, and `scripts/register_model.py` looks runs up by name and refuses to continue unless exactly one finished run matches. On 2026-11-08 I deleted the duplicates (an MLflow soft delete) and kept the runs that the registry versions point to.

## 3.6 Model registry

### Registering and aliasing

`scripts/register_model.py` takes the run names to register plus `--champion` and `--challenger`. For each run it registers the run's logged model, waits until the version is READY, and tags the version with val/test ROC-AUC, val/test F1, the run name and the threshold. It also writes a one-line description. Then it points the aliases at the right versions. The registered model `23L-0584_A2_Ops_visionguard` has three versions (Figure 3.22): v1 = E0-base, v2 = E7-ft-lr-1e-4 (alias `challenger`), v3 = E5-mobilenetv2 (alias `champion`). Figure 3.23 shows the version-3 page with its source run and description, and Figure 3.24 shows its tags. I used aliases rather than fixed version numbers so that consumers ask for "the champion". Promoting a different model then means moving the alias, not changing code.

### Load by alias

`scripts/predict_champion.py` is a separate, small script. It resolves the alias with `get_model_version_by_alias`, downloads that run's `serving_config.json`, and loads `models:/23L-0584_A2_Ops_visionguard@champion`. It then preprocesses the image with the same `decode_and_resize` used in training and applies the stored threshold. In Figure 3.25, `RealArt/images37.jpg` → real (p_ai 0.021572, confidence 0.978) and `AiArtData/sddefault.jpg` → ai (p_ai 0.733, threshold 0.6725). Loading took 10.5–12.0 s, while predicting took 0.71 s.

### How the champion reaches the server

The server never calls DagsHub per request. `scripts/export_champion.py` resolves the `champion` alias once and downloads that run's `keras_model/model.keras` as `models/champion.keras`. It saves `models/serving_config.json` with the threshold, image size, resize mode, version, alias, run id and export time, then test-loads the model and prints its SHA-256. Both files are git-ignored. The app's `ChampionModel` (`src/visionguard/inference.py`) loads them once at start-up from `MODEL_PATH`. A new champion only needs the alias moved, the export re-run and the server restarted. The load times above show why a per-request download would be unacceptable.

![Figure 3.22 — Registered model `23L-0584_A2_Ops_visionguard`: v1 = E0-base, v2 = E7-ft-lr-1e-4 (alias challenger), v3 = E5-mobilenetv2 (alias champion), each tagged with run name, val/test F1 and AUC, and threshold.](../reports/screenshots/task3/3.6_registry_versions_aliases.png)

![Figure 3.23 — Version 3 page: source run E5-mobilenetv2, alias champion, description with val/test ROC-AUC and threshold 0.6725.](../reports/screenshots/task3/3.6_registry_version3_champion.png)

![Figure 3.24 — Version 3 tags: test_f1, test_roc_auc, threshold, val_f1, val_roc_auc.](../reports/screenshots/task3/3.6_registry_version3_tags.png)

![Figure 3.25 — `scripts/predict_champion.py` loads `models:/23L-0584_A2_Ops_visionguard@champion` (v3) from DagsHub: RealArt/images37.jpg → real (confidence 0.978), AiArtData/sddefault.jpg → ai (0.733, threshold 0.6725). Loading takes ~10–12 s vs ~0.7 s to predict.](../reports/screenshots/task3/3.6_predict_by_alias_terminal.png)

### What went wrong

My first attempt registered `runs:/<run_id>/model`, and DagsHub could not resolve that URI. In MLflow 3, a logged model is a separate "LoggedModel" entity with its own id. So `register_model.py` now looks up the run's READY logged model with `mlflow.search_logged_models` and registers `models:/<model_id>`, which fixed it.
