#!/usr/bin/env bash
# Task 1.4 (apply) evidence on one screen: run from the project root in YOUR terminal, then screenshot it.
# Afterwards it opens the two DagsHub pages to screenshot in your browser (address bar visible).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"

run() { printf '\033[1;33m$ %s\033[0m\n' "$*"; "$@"; echo; }

clear 2>/dev/null || true
printf '\033[1;36m══ Task 1.4 apply: dataset versioned with DVC + DagsHub ══\033[0m  %s@%s  %s\n\n' \
  "$(whoami)" "$(hostname)" "$(date '+%F %T')"
run git --no-pager log --oneline --decorate -8
run cat data/raw.dvc            # pointer committed to Git: md5 of the directory, size, nfiles
run cat data/.gitignore         # DVC keeps the 504 MB of images out of Git
run git check-ignore -v data/raw/RealArt/images37.jpg .dvc/config.local
run dvc remote list
run dvc status -c               # local cache and DagsHub storage in sync

if [[ "${OPEN_BROWSER:-1}" == 1 ]] && command -v xdg-open >/dev/null; then
  xdg-open "https://dagshub.com/MHassanAbaid24/23L-0584_A2_Ops/src/main/data" >/dev/null 2>&1 || true
  xdg-open "https://dagshub.com/MHassanAbaid24/23L-0584_A2_Ops_dvc_toy" >/dev/null 2>&1 || true
  echo "Opened in your browser: project data/ folder (raw is DVC-tracked) and the toy repo."
fi
