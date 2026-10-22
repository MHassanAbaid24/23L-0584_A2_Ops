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

### Still missing for Task 1

| Needed | Sub-task | How to capture |
|---|---|---|
| Toy repo on DagsHub (web) | 1.4 toy "commit both versions (dagshub)" | Browser: `https://dagshub.com/MHassanAbaid24/23L-0584_A2_Ops_dvc_toy` → Commits tab and `toy.csv` with its DVC badge; address bar visible. |
| Project DVC apply (terminal) | 1.4 apply | `bash labs/task1_evidence.sh` → screenshot the terminal (save as `task1/1.4_apply_01_project_dvc.png`). |
| Project dataset on DagsHub (web) | 1.4 apply | Browser: `https://dagshub.com/MHassanAbaid24/23L-0584_A2_Ops` → `data/raw` folder showing it is DVC-tracked (975 files). |
| Audit summary (optional) | 1.2 | `sed -n 1,30p reports/audit/summary.md`. The report can use the generated table directly. |

To avoid the pager problem in later screenshots, use `git --no-pager …`. The toy script has been updated to do this.
