#!/usr/bin/env bash
# Task 1.4 (toy part) — DVC basics on a small CSV, pushed to DagsHub.
#
# Run from the project root in YOUR terminal so screenshots show your prompt:
#     bash labs/dvc_toy_lab.sh            # pauses after each step for a screenshot
#     PAUSE=0 bash labs/dvc_toy_lab.sh    # no pauses
#
# Needs DAGSHUB_USER and DAGSHUB_TOKEN in ./.env and the DagsHub repo
# <DAGSHUB_USER>/23L-0584_A2_Ops_dvc_toy (already created, empty).
# The token is only passed through git's credential helper and DVC's git-ignored
# config.local, so it never appears on screen or in a commit.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TOY_DIR="${TOY_DIR:-$PROJECT_DIR/../dvc-toy}"
PAUSE="${PAUSE:-1}"
export PATH="$PROJECT_DIR/.venv/bin:$PATH"
set -a; . "$PROJECT_DIR/.env"; set +a
REPO="${TOY_REPO:-23L-0584_A2_Ops_dvc_toy}"
GIT_URL="https://dagshub.com/$DAGSHUB_USER/$REPO.git"
DVC_URL="https://dagshub.com/$DAGSHUB_USER/$REPO.dvc"
CRED_HELPER='!f() { echo "username=$DAGSHUB_USER"; echo "password=$DAGSHUB_TOKEN"; }; f'

step() { printf '\n\033[1;36m══ %s ══\033[0m  (%s)\n' "$1" "$(date '+%F %T')"; }
run()  { printf '\033[1;33m$ %s\033[0m\n' "$*"; "$@"; }
pause() { [[ "$PAUSE" == 1 ]] && read -rp $'\n[screenshot now, then press Enter] ' _ || true; }
gitpush() { printf '\033[1;33m$ git push %s\033[0m\n' "$*"; git -c credential.helper= -c "credential.helper=$CRED_HELPER" push "$@"; }

if [[ -e "$TOY_DIR" ]]; then
  echo "$TOY_DIR already exists; delete it first for a clean run (rm -rf \"$TOY_DIR\")."; exit 1
fi

step "1. New empty folder; initialise Git and DVC"
run mkdir -p "$TOY_DIR"; cd "$TOY_DIR"
run git init -b main
run dvc init
run git commit -m "Initialise DVC"
run git status --short
pause

step "2. Version 1 of the CSV, tracked by DVC (not Git)"
printf 'id,image,label\n1,cat.jpg,real\n2,robot.png,ai\n3,dog.jpg,real\n' > toy.csv
run cat toy.csv
run dvc add toy.csv
run cat toy.csv.dvc          # the small pointer file: md5 hash, size, path
run cat .gitignore           # DVC told Git to ignore the real data file
run git add toy.csv.dvc .gitignore
run git commit -m "Track toy.csv v1 with DVC"
pause

step "3. Change the data -> version 2, commit again"
printf '4,castle.png,ai\n5,beach.jpg,real\n' >> toy.csv
run cat toy.csv
run dvc add toy.csv
run git diff toy.csv.dvc     # only the hash and size in the pointer changed
run git commit -am "Update toy.csv to v2 (5 rows)"
run git log --oneline
pause

step "4. Return to the older version (v1), then back to v2"
run git checkout HEAD~1 -- toy.csv.dvc   # old pointer from Git...
run dvc checkout toy.csv                 # ...DVC restores matching bytes from its cache
run cat toy.csv                          # 3 rows again
run git checkout HEAD -- toy.csv.dvc
run dvc checkout toy.csv
run cat toy.csv                          # back to v2, 5 rows
pause

step "5. Configure the DagsHub remotes (Git + DVC); credentials stay in .dvc/config.local"
run git remote add origin "$GIT_URL"
run dvc remote add -d origin "$DVC_URL"
dvc remote modify origin --local auth basic
dvc remote modify origin --local user "$DAGSHUB_USER"
dvc remote modify origin --local password "$DAGSHUB_TOKEN"
echo "(auth written to .dvc/config.local, which DVC git-ignores)"
run cat .dvc/config
run git commit -am "Configure DagsHub DVC remote"
pause

step "6. Push: commits to Git, data of BOTH versions to DVC storage"
gitpush -u origin main
run dvc push                             # v2 (current pointer)
# 'dvc push --all-commits' would read each commit's own .dvc/config, and the v1/v2 commits
# predate the remote, so it silently skips them. Push v1 explicitly instead:
run git checkout HEAD~2 -- toy.csv.dvc
run dvc push                             # v1
run git checkout HEAD -- toy.csv.dvc
run dvc status -c                        # cache and remote in sync
pause

step "7. Delete the local data AND cache, then pull it back from DagsHub"
run rm toy.csv
run rm -rf .dvc/cache
run ls -la
run dvc pull
run cat toy.csv                          # v2 restored from DagsHub
run git checkout HEAD~2 -- toy.csv.dvc
run dvc pull
run cat toy.csv                          # v1 also restored from DagsHub
run git checkout HEAD -- toy.csv.dvc
run dvc checkout toy.csv
pause

step "Done — see https://dagshub.com/$DAGSHUB_USER/$REPO"
run git log --oneline --graph
run git status --short                   # clean: nothing left uncommitted
