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
