# Screenshot evidence index

Naming: `<subtask>_<topic>_<step>_<what it shows>.png`, one folder per task. The brief requires every screenshot
to show context (prompt with username, date/time, project name). The **Proof** column records which of those
each screenshot shows; ✅ = usable as is, ⚠️ = usable but weak, see note.

## Task 1

| File | Sub-task | Suggested caption | Proof | Assessment |
|---|---|---|---|---|
| `task1/1.1_env_versions_gpu_check.png` | 1.1 | Exact library versions in the project venv and TensorFlow's GPU check (0 GPUs: no NVIDIA driver loaded, CPU only). | prompt, project, timestamp 2026-10-22 14:43 | ✅ |
| `task1/1.4_toy_01_init_git_dvc.png` | 1.4 toy | New empty folder `dvc-toy`: `git init`, `dvc init`, first commit `ae58d01`. | prompt, project, timestamp 14:44:55 | ✅ A "Follow link" tooltip covers part of DVC's analytics banner (irrelevant text). |
| `task1/1.4_toy_02_track_v1_pointer_file.png` | 1.4 toy | v1 of `toy.csv` tracked with `dvc add`; the pointer file (md5 `00cd48…`, 60 bytes) and the `.gitignore` entry DVC created; commit `9ffec16`. | timestamp 14:45:39 | ✅ Best image for explaining the pointer file. |
| `task1/1.4_toy_03a_change_to_v2_commit.png` | 1.4 toy | Data changed to v2 (5 rows), re-added and committed as `ee5c6fc`. | timestamp 14:45:57 | ⚠️ A file-path tooltip covers the `$ dvc add toy.csv` line. |
| `task1/1.4_toy_03b_pointer_diff_v1_v2.png` | 1.4 toy | `git diff` of the pointer: between versions only the md5 and size change. | none (inside git's pager) | ⚠️ No prompt or time; pair it with 03a in the report. |
| `task1/1.4_toy_03c_log_both_versions.png` | 1.4 toy | Git history with both data versions committed. | none (pager) | ⚠️ Same as 03b; 08 shows the same commits and more. Optional. |
| `task1/1.4_toy_04_return_to_v1_and_back.png` | 1.4 toy | Return to v1: check out the old pointer, `dvc checkout` restores the 3-row file; then back to v2. | timestamp 14:47:00 | ✅ |
| `task1/1.4_toy_05_configure_dagshub_remote.png` | 1.4 toy | DagsHub Git + DVC remotes configured; `.dvc/config` holds only the URL (credentials go to the git-ignored `config.local`). | timestamp 14:47:14 | ✅ Shows that no secret is committed. |
| `task1/1.4_toy_06_push_both_versions.png` | 1.4 toy | Commits pushed to DagsHub Git; data of v2 and then v1 pushed to DagsHub storage; cache and remote in sync. | timestamp 14:47:30 | ✅ |
| `task1/1.4_toy_07a_delete_data_cache_pull.png` | 1.4 toy | Data file and the whole local cache deleted (`ls -la` proves it), then `dvc pull` restores v2 from DagsHub, and v1 after checking out the old pointer. | timestamp 14:47:53 | ✅ Key evidence for "delete local data and pull back". |
| `task1/1.4_toy_07b_pull_v1_restore_v2.png` | 1.4 toy | Continuation: v1 pulled from the remote, then the workspace returned to v2. | none (scrolled) | ⚠️ Mostly overlaps 07a; use only if 07a is cropped. |
| `task1/1.4_toy_08_final_commit_graph.png` | 1.4 toy | Final history: 4 commits, `main` in sync with `origin/main` on DagsHub. | none (pager) | ⚠️ No prompt or time. |
| `task1/1.4_toy_09_dagshub_repo.png` | 1.4 toy | The toy repo on DagsHub: 4 commits pushed, `toy.csv` shown with a `dvc` badge (data in DagsHub storage, only `toy.csv.dvc` in Git). | address bar `dagshub.com/MHassanAbaid24/23L-0584_A2_Ops_dvc_toy`, clock Oct 22 16:15 | ✅ |
| `task1/1.4_apply_01_project_dvc.png` | 1.4 apply | The project's Git history, the `data/raw.dvc` pointer (md5, 503,783,378 bytes, 975 files), DVC's `data/.gitignore`, proof that images and `config.local` are git-ignored, the DagsHub remote, and cache ↔ remote in sync. | user@host, timestamp 16:12:12 | ✅ Main evidence for 1.4 apply. |
| `task1/1.4_apply_02_dagshub_data_folder.png` | 1.4 apply | DagsHub project `23L-0584_A2_Ops`, connected to GitHub: `data/raw` carries the `dvc` badge, while Git holds only `raw.dvc` and `.gitignore`. | address bar `…/23L-0584_A2_Ops/src/main/data`, clock Oct 22 16:15 | ✅ |

### Still missing / to improve for Task 1

| Item | Sub-task | Action |
|---|---|---|
| Audit summary (optional) | 1.2 | `sed -n 1,30p reports/audit/summary.md`. The report can use the generated table directly. |

To avoid the pager problem in later screenshots, use `git --no-pager …`. The toy script has been updated to do this.

## Task 2

| File | Sub-task | Suggested caption | Proof | Assessment |
|---|---|---|---|---|
| `task2/2.2_cnn_model_summary_1.png` | 2.2 | Baseline CNN summary, part 1: input, rescaling and the first conv blocks with output shapes and parameter counts. | VS Code terminal prompt, clock 18:07 | ✅ |
| `task2/2.2_cnn_model_summary_2.png` | 2.2 | Part 2: the last conv blocks, GAP, dropout, sigmoid output; total 390,113 parameters (389,153 trainable). | prompt, clock 18:08 | ✅ |

## Task 3

| File | Sub-task | Suggested caption | Proof | Assessment |
|---|---|---|---|---|
| `task3/3.1_toy_two_local_runs_terminal.png` | 3.1 | `labs/mlflow_toy.py --target local`: two runs, lr 0.001 (val acc 0.8965) and lr 0.01 (0.907), with their run ids. | prompt, clock 18:09 | ✅ |
| `task3/3.1_local_ui_runs_list.png` | 3.1 | Local MLflow UI, experiment `toy-mnist`: both runs, each with a logged model. | address bar 127.0.0.1:5000, clock 18:12 | ✅ |
| `task3/3.1_local_run_lr0.01_overview.png` | 3.1 | Run `lr-0.01`: 6 parameters, 4 final metrics, tags `lab`, `target`, `author`; source `mlflow_toy.py` at commit d133853. | address bar, clock 18:13 | ✅ Good for the parameter/metric/tag definitions. |
| `task3/3.1_local_run_lr0.01_artifact_curves.png` | 3.1 | Artifact `curves.png`: train/val loss and accuracy per epoch; validation loss rises after epoch 4 (overfitting at lr 0.01). | address bar, clock 18:13 | ✅ |
| `task3/3.1_local_compare_plot.png` | 3.1 | Compare view: parallel-coordinates plot of learning rate vs final training loss. | address bar, clock 18:14 | ✅ |
| `task3/3.1_local_compare_params_metrics.png` | 3.1 | Compare view: only `learning_rate` differs (highlighted); lr 0.01 reaches val acc 0.907 vs 0.896 and val loss 0.332 vs 0.34. | address bar, clock 18:14 | ✅ |
| `task3/3.2_env_gitignored_proof.png` | 3.2 | `git check-ignore -v .env` → matched by `.gitignore` line 2; `git status` is clean, so the credentials file has never been committed; `.env.example` (7 lines, no values) is the committed template. | VS Code prompt, clock 18:15 | ✅ The `.env` tab is open but its contents are not visible. |
| `task3/3.2_dagshub_experiments_tab.png` | 3.2 | DagsHub project Experiments tab (22 runs), connected to GitHub: the toy runs `lr-0.001`/`lr-0.01` plus all VisionGuard runs, each linked to its Git commit. | address bar, clock 18:16 | ✅ Also usable as an overview for 3.5. |
| `task3/3.2_dagshub_mlflow_toy_runs.png` | 3.2 | The same toy run on DagsHub's hosted MLflow (`…23L-0584_A2_Ops.mlflow`), experiment `toy-mnist`: both runs with logged models. | address bar, clock 18:17 | ✅ |
| `task3/3.3_E0_manual_params.png` | 3.3 | Manual logging (E0-base): parameters describe the experiment: model, backbone, img_size, lr, fine_tune_lr, epochs per stage, dropout, augment, resize_mode, patience, threshold_beta (30 in total). | address bar (DagsHub MLflow), clock 18:18 | ✅ |
| `task3/3.3_E0_manual_metrics_tags.png` | 3.3 | E0-base: 39 metrics (train/val/test), reproducibility tags git_commit, git_dirty=False, seed 42, dvc_data_md5, dvc_data_nfiles 975, logging_mode=manual; source train.py at 8b73f44; registered as model v1. | address bar, clock 18:18 | ✅ Key evidence for the tags. |
| `task3/3.3_E0_manual_metric_curves.png` | 3.3 | E0-base per-epoch curves: val_loss over all 35 epochs (both stages), val_precision per epoch; one-off values (val_fp) as bars. | address bar, clock 18:19 | ✅ |
| `task3/3.3_E1_autolog_params.png` | 3.3 | Autolog (E1-base-autolog): 31 parameters, but they are fit()/optimizer settings (batch_size None, validation_split, shuffle, class_weight, opt_name adam, opt_learning_rate 0.001); epochs=20 and initial_epoch=0 come from the first fit() call only. Nothing about the backbone, image size or augmentation. | address bar, clock 18:19 | ✅ |
| `task3/3.3_E1_autolog_metric_curves.png` | 3.3 | E1-base-autolog curves: val_loss stops at step 19 (20 points); the 15 fine-tuning epochs are missing because autolog restarts the step counter on the second fit() call. Compare with the 35 points in E0. | address bar, clock 18:20 | ✅ Key evidence for "what autolog misses". |
| `task3/3.3_E1_autolog_metrics_tags.png` | 3.3 | E1-base-autolog: the same reproducibility tags (git_commit, git_dirty, seed, dvc_data_md5, dvc_data_nfiles, logging_mode=autolog, threshold), which come from the training code, not from autolog. Autolog's `learning_rate` metric is stale at 0.001 (the stage-1 value), while E0's manual callback logs the real final 1e-5. Not registered. | address bar, clock 18:20 | ✅ |
| `task3/3.4_E5_artifacts_tree.png` | 3.4 | Champion run E5-mobilenetv2, Artifacts: `figures/` (val/test confusion matrices, ROC, curves), `keras_model/model.keras`, `run_info/` (requirements.txt, metrics, serving_config, model summary, layer table, trainable layers, history), `splits/` (train/val/test CSVs), stored in DagsHub's MLflow artifact store. | address bar, clock 18:21 | ✅ |
| `task3/3.4_E5_artifact_confusion_matrix_test.png` | 3.4 | Test confusion matrix at threshold 0.673: 60 TN, 6 FP (real called AI), 26 FN, 55 TP. | address bar, clock 18:22 | ✅ |
| `task3/3.4_E5_artifact_requirements.png` | 3.4 | `run_info/requirements.txt`: the full `pip freeze` of the Kaggle training environment (18 KB), so the run can be reproduced exactly. | address bar, clock 18:22 | ✅ It includes Kaggle's preinstalled packages; the serving image will pin a slim list instead. |
| `task3/3.5_compare_scatter_auc_vs_real_called_fake.png` | 3.5 | Compare view of E0–E7: validation ROC-AUC (x) vs rate of real images called AI (y). The point at (0.881, 0.092), bottom-right and alone, is E5-mobilenetv2: best ranking quality and fewest false accusations. | address bar, clock 18:24 | ✅ Main chart for the justification. |
| `task3/3.5_compare_params_table.png` | 3.5 | Compare view, parameter table: backbone, augment and fine_tune_epochs differ between the runs (run names are above, out of view). | address bar, clock 18:25 | ⚠️ Supplementary only. |
| `task3/3.5_compare_parallel_coordinates.png` | 3.5 | Parallel coordinates for E0–E7: backbone, img_size, augment, resize_mode, fine_tune_lr, fine_tune_epochs → val ROC-AUC (colour). Red, high-AUC lines run through MobileNetV2 and 224 px; the dark-blue low line is img_size 160 (E4). The line marked "unknown" is E1 (autolog logs no experiment parameters). | address bar, clock 18:27 | ✅ |
| `task3/3.5_runs_table_sorted_by_val_auc.png` | 3.5 | Runs table sorted by validation ROC-AUC, with test F1/AUC, val F1, val real→fake rate, backbone and img size: E5-mobilenetv2 leads on val AUC (0.881) with the lowest real→fake rate (0.092); cnn-baseline is last (0.696). The CNN's `backbone` cell is the unused default. | address bar, clock 18:30 | ✅ Main table for the justification. |
| `task3/3.6_registry_versions_aliases.png` | 3.6 | Registered model `23L-0584_A2_Ops_visionguard`: v1 = E0-base, v2 = E7-ft-lr-1e-4 (alias **challenger**), v3 = E5-mobilenetv2 (alias **champion**), each tagged with run name, val/test F1 and AUC, and threshold. | address bar, clock 18:31 | ✅ |
| `task3/3.6_registry_version3_champion.png` | 3.6 | Version 3 page: source run E5-mobilenetv2, alias champion, description with val/test ROC-AUC and threshold 0.6725. | address bar, clock 18:32 | ✅ |
| `task3/3.6_registry_version3_tags.png` | 3.6 | Version 3 tags scrolled: test_f1, test_roc_auc, threshold, val_f1, val_roc_auc. | address bar, clock 18:32 | ✅ Complements the previous one. |
| `task3/3.6_predict_by_alias_terminal.png` | 3.6 | `scripts/predict_champion.py` loads `models:/23L-0584_A2_Ops_visionguard@champion` (v3) from DagsHub: RealArt/images37.jpg → real (confidence 0.978), AiArtData/sddefault.jpg → ai (0.733, threshold 0.6725). Loading takes ~10–12 s vs ~0.7 s to predict. | VS Code prompt, clock 18:33 | ✅ The load-time vs predict-time gap is the argument for loading once at start-up (4.1). |

## Task 4

Captured with a headless Chromium driven against the real app (gunicorn, debug off, champion v3, local SQLite) on
Sun 08 Nov 2026 12:40–12:43 PKT. A headless browser has no address bar or clock, so a strip at the top of each image
states the real URL, capture time, user@host and project. Terminal items are rendered transcripts of commands actually
run on this machine at the time shown. Say this once in the report's front matter.

| File | Sub-task | Suggested caption | Proof | Assessment |
|---|---|---|---|---|
| `task4/4.1_inference_console_two_known_images.png` | 4.1 | `python -m visionguard.inference` loads the champion once (9.1 s cold start), then classifies two known images in ~0.3 s each: RealArt/images37.jpg → real (0.978), AiArtData/sddefault.jpg → ai (0.733). | prompt, `date` 12:42:39 PKT | ✅ The load vs predict gap shows why loading per request is unacceptable. |
| `task4/4.2_reject_wrong_type_tiff.png` | 4.2 | A real photo saved as TIFF is rejected with **415 Unsupported Media Type**: the format is read from the bytes, and TIFF is not allowed. | URL strip, 12:42:04 | ✅ Failure case 1 (wrong type). |
| `task4/4.2_reject_too_large_17mb.png` | 4.2 | A 17 MB PNG is rejected with **413** before the body is read (limit 10 MB from `MAX_UPLOAD_MB`). | URL strip, 12:42:07 | ✅ Failure case 2 (size). |
| `task4/4.2_reject_undecodable_fake_jpg.png` | 4.2 | A text file named `vacation.jpg` is rejected with **400**: "looks like an image but cannot be decoded". | URL strip, 12:42:10 | ✅ Failure case 3 (undecodable). |
| `task4/4.3_upload_page.png` | 4.3 | Upload page: file picker limited to JPEG/PNG/WEBP/GIF/BMP up to 10 MB; Upload / History / Health menu. | URL strip, 12:40:57 | ✅ |
| `task4/4.3_result_real_correct.png` | 4.3 | A real test photo → "Likely real", 99.6% confidence, P(AI) 0.004, threshold 0.673, model v3, honest caveat. | URL strip, 12:41:35 | ✅ |
| `task4/4.3_result_ai_correct.png` | 4.3 | A Midjourney collage → "Likely AI-generated", 99.8%. | URL strip, 12:41:38 | ✅ |
| `task4/4.3_result_ai_borderline.png` | 4.3 | An AI image just above the threshold → AI-generated at 68.5% (P(AI) 0.685 vs 0.673). Shows what a low-confidence verdict looks like. | URL strip, 12:41:47 | ✅ |
| `task4/4.3_result_ai_missed.png` | 4.3 | An AI image the model **misses** → "Likely real" 92.4%. The caveat warns about exactly this kind of photorealistic fake. | URL strip, 12:41:50 | ✅ Good for an honest discussion; also in the 2.6 error analysis. |
| `task4/4.4_health_check_json.png` | 4.4 | `/health`: status ok, database ok, `debug: false`, model v3 from run E5-mobilenetv2, threshold 0.6725, 224 px, loaded in 1.97 s. | URL strip, 12:42:22 | ✅ |

## Task 5

| File | Sub-task | Suggested caption | Proof | Assessment |
|---|---|---|---|---|
| `task5/5_history_page_via_menu.png` | 5 | History opened from the menu: 7 predictions (3 AI, 4 real), newest first, each with UTC time, file, decision, confidence, P(AI), threshold, model + version, latency and source. | URL strip, 12:42:20 | ✅ Answers "what, when, which model, how confidently". |
| `task5/5_history_record_7_detail.png` | 5 | Clicking id 7 in history reopens that stored record at `/result/7`. | URL strip, 12:42:22 | ✅ |
| `task5/5_db_query_predictions_table.png` | 5 | Direct `sqlite3` query of the `predictions` table, its schema (indexes on label and created_at), and `ls instance/` showing only the database: no uploaded images on disk. | prompt, `date` 12:42:50 PKT | ✅ Row 1 duplicates row 2: a real repeated upload from the capture session. |
