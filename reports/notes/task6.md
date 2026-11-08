# Task 6 — facts, decisions and problems (source for the report)

Evidence transcripts: `reports/evidence/task6/*.txt` → rendered screenshots `reports/screenshots/task6/*.png`
(rendered with `scripts/render_transcripts.py`; each image is labelled as a rendered transcript of real command output).
Files: `Dockerfile`, `docker/Dockerfile.before`, `.dockerignore`, `requirements-serve.txt`, `compose.yaml`, `.env.example`,
`.github/workflows/ci.yml`, `tests/`, README badge. Warm-up definitions: `reports/notes/task6_1_definitions.md`.

## 6.1 Warm-up (nginx:alpine)
Six transcripts: pull + image history (layers), run with `-p 8081:80` and a read-only bind mount (curl shows the custom
`<h1>23L-0584_A2_Ops docker warm-up</h1>`), live edit on the host visible without restart, a named volume written and read
by two throw-away alpine containers, `docker logs`, `docker exec` shell (root inside nginx image), stop/rm, volume rm, rmi.
Problem: the first `docker pull` hit a TLS handshake timeout; a retry worked.

## 6.2 Dockerfile
- Base `python:3.12-slim` (glibc; TensorFlow wheels do not support Alpine/musl). Order: base → user → requirements file →
  pip install → app code. Source layer is 67.7 kB; the dependency layer is 1.61 GB, so a code edit rebuilds in seconds.
- `tensorflow-cpu==2.20.0` (wheel 259 MB) instead of `tensorflow` (621 MB, CUDA-capable): the server has no GPU.
  Training extras (mlflow, keras-tuner, dvc, sklearn, matplotlib) are not in `requirements-serve.txt`.
- Non-root user `app` uid 10001 (`whoami` → app). HEALTHCHECK calls /health with Python's urllib (no curl in slim image),
  interval 30 s, start period 90 s (model load). `.dockerignore` excludes .env, .git, .venv, data, models, reports, …
  → no secrets and no model inside the image (`ls -a /app` shows only gunicorn.conf.py, instance, requirements-serve.txt, src).
- ENV: PYTHONPATH=/app/src (no packaging step), KERAS_HOME=/tmp/.keras (home dir not writable by `app`), PIP timeouts.
- Image: 1.73 GB (1,728,429,175 bytes), TensorFlow 2.20.0 / Keras 3.15.1, GPUs [].
Problems: (1) the first build failed after 14 min: PyPI read timeout while downloading the TensorFlow wheel → added
PIP_DEFAULT_TIMEOUT=120 and PIP_RETRIES=10; the retry built in 3 min 43 s. (2) This machine's Docker has no buildx
plugin, so `--progress=plain` was rejected and the legacy builder was used locally; CI uses BuildKit via
docker/setup-buildx-action. (3) The first evidence command used nested shell quoting that silently broke the TF version
line; re-recorded with plain `--entrypoint` commands.

## 6.3 Compose
- Services `web` (build ., image 23l-0584-a2-ops-web:latest, port 8000) and `db` (postgres:16-alpine, named volume
  `pgdata`, healthcheck `pg_isready`, interval 5 s). `web.depends_on.db.condition: service_healthy`.
- web connects with `DATABASE_URL=postgresql+psycopg://visionguard:${POSTGRES_PASSWORD}@db:5432/visionguard` → reaches the
  database by service name on the project network (172.22.0.2 inside `vg-23l-0584_default`).
- No `ports:` for db: `docker compose port db 5432` → `:0`; `PortBindings={}`; only `ExposedPorts 5432/tcp`.
  Note: on this host port 5432 is published by an unrelated container (`votesphere-db`, another project), so "connect to
  localhost:5432" would have proved nothing; the evidence shows our db has no binding and only web has 8000.
- Secrets: POSTGRES_PASSWORD and SECRET_KEY live in the untracked `.env` (generated with secrets.token_hex);
  `.env.example` documents them; `${VAR:?message}` makes Compose refuse to start if they are missing. Each service gets
  only the variables listed under `environment` (the DagsHub tokens in .env never reach a container).
- Model: mounted read-only `./models:/app/models:ro`, not baked in. Why: a new champion is deployed by re-exporting the
  file and restarting (no image rebuild); the image stays generic and CI can build it without the model (models/ is not in
  Git); 25 MB less per image version. Trade-off: the host must have the file; a missing file is caught at start-up
  (the app loads the model once and fails fast).
- Result: `docker compose ps` → db Up (healthy) 5432/tcp; web Up (healthy) 0.0.0.0:8000->8000/tcp; /health → database ok,
  debug false, model v3.

## 6.4 Image analysis
- Largest layer: pip install 1.61 GB; inside it tensorflow 1.2 GB, pandas 72 MB, clang 62 MB, numpy 43 MB, keras 30 MB.
- Optimisation applied: `--no-cache-dir`. Before (docker/Dockerfile.before, the only difference): 2,097,435,883 bytes
  (2.1 GB), pip layer 1.98 GB, `/root/.cache/pip` = 355 MB. After: 1,728,429,175 bytes (1.73 GB), pip layer 1.61 GB, no
  cache. Saving 369 MB (17.6 %). Conclusion: the cache is pure dead weight in an image; the remaining size is TensorFlow
  itself — the next step would be TFLite at serving time (Task 7 does this on Vercel: 8.9 MB model + 21.8 MB runtime).

## 6.5 Operations and troubleshooting
- stop/start web, restart db, ps, logs of both services, shell in both (`exec web sh` → user app, models mounted;
  `exec db psql` → the predictions table with a row written by the containerised app, source local).
- Web survived the db restart without restarting itself: SQLAlchemy `pool_pre_ping=True` replaces dead connections.
- Break: `env POSTGRES_PASSWORD=not-the-real-password docker compose up -d --no-deps web` (only web recreated; db keeps
  the real password because PostgreSQL stores it in the volume at first init). Symptom: `Restarting (3)`.
  Diagnosis only from logs: web log `psycopg.OperationalError … FATAL: password authentication failed for user
  "visionguard"` → gunicorn `Worker failed to boot`; db log shows the same FATAL with the pg_hba rule; `psql` inside db
  works → db is fine, the client credential is wrong. Fix: `docker compose up -d --no-deps web` (back to .env value) →
  healthy, /health ok.

## 6.6 CI
- Toy: public repo https://github.com/MHassanAbaid24/23L-0584_A2_Ops_ci_toy, matrix Python 3.11 + 3.12, on push.
  Red run 42465926045 (deliberate `assert add(2, 2) == 5` → `assert 4 == 5`, both legs failed), green run 42466069364
  after the fix commit "Fix deliberately failing test".
- Project: `.github/workflows/ci.yml` → job `test` (Python 3.12, pip cache, install requirements-serve.txt + pytest + ruff,
  `pip install --no-deps -e .`, `ruff check src scripts labs tests`, `pytest -q`) then job `docker` (buildx, build only,
  GitHub Actions layer cache). First run 42467537562 green: test 52 s, docker 2 min 12 s; `14 passed, 2 warnings in
  2.69s`. Badge at the top of README.md. Later commits add tests/test_remote.py (18 tests in total locally).
- Tests (tests/): preprocessing (shape/dtype/range, grey/RGBA/palette → RGB, pad vs stretch, content-based decode),
  validation (TIFF 415, text 400, truncated JPEG 400, WEBP ok), Flask app with an injected fake predictor and temp SQLite
  (rows written only on success, sanitised filename, history order, 404, health), a real tiny saved Keras model through
  ChampionModel, and the remote client against a local stand-in API. Mutation check: forcing 415→400 made
  `test_failed_uploads_do_not_create_prediction_records` fail (`assert 400 == 415`).
- Secrets: none are needed by this workflow. Any future secret goes in GitHub → Settings → Secrets and variables →
  Actions and is referenced as `${{ secrets.NAME }}`: the workflow file is in Git (and visible to anyone who can read the
  repo or its forks), while Actions secrets are encrypted, injected only at run time and masked in logs.
- Warnings shown by GitHub: Node.js 20 actions are being forced to Node 24 (checkout@v4, setup-python@v5, buildx@v3,
  build-push@v6) — harmless now; bump action versions later.
