# Tasks 4 & 5 — draft notes for the report (rewrite in your own words)

Code: `src/visionguard/inference.py` (4.1), `src/visionguard/web/` (app factory + routes, `db.py`, `config.py`,
`templates/`, `static/style.css`), `gunicorn.conf.py`.

## 4.1 Inference module

- `ChampionModel` loads `models/champion.keras` and `serving_config.json` once, when the process starts, and runs
  one warm-up prediction. All requests reuse that object.
- Why loading per request is unacceptable (measured): loading takes 2.5–6 s locally (10–12 s through the MLflow
  registry), while a prediction takes 0.1–0.2 s. Per-request loading would make every request 20–50× slower, use
  several hundred MB of memory per concurrent request, and hit DagsHub on every upload.
- Preprocessing: serving calls the same `visionguard.data.decode_and_resize` used in training (decode by
  content → 3-channel RGB → antialiased bilinear resize to 224×224 → uint8 → float32 [0, 255]; the model normalises
  internally). Proof: the p_ai values from the exported file (0.0216 and 0.7333) are identical to the
  MLflow-registry model in 3.6. If training and serving preprocessed differently (e.g. a different resize, BGR
  order, or [0,1] scaling), the model would see inputs it was never trained on and its accuracy would silently drop.
- Console output for two known images: `python -m visionguard.inference data/raw/RealArt/images37.jpg data/raw/AiArtData/sddefault.jpg`
  → real (p_ai 0.022, confidence 0.978) and ai (p_ai 0.733, confidence 0.733).
- WEBP is not decodable by `tf.io.decode_image`, so WEBP uploads are first converted losslessly to PNG (identical pixels).

## 4.2 Routes and validation

| Route | Purpose |
|---|---|
| `GET /` | upload form |
| `POST /predict` | validate → predict → store a row → render the result page |
| `GET /result/<id>` | re-open a stored verdict (without the image, which is never stored) |
| `GET /history` | paginated trail from the database, newest first |
| `GET /health` | JSON: model name/version/threshold, database reachability, debug flag; HTTP 503 if the DB is down |

Validation (verified with curl):
- Wrong type → **415**. A real TIFF is a valid image but not an allowed format. The format is read from the file's
  magic bytes by Pillow, never from the extension or the browser's MIME type.
- Too large → **413** via Flask `MAX_CONTENT_LENGTH` (10 MB by default, from `MAX_UPLOAD_MB`). The request is
  rejected before the body is read into memory.
- Looks like an image but cannot be decoded → **400**. A text file renamed `.jpg` and a truncated JPEG both fail.
  Pillow `verify()` checks the structure, then a full `load()` decodes every pixel, which catches truncated files.
- Decompression bombs (>100 MP) → 400.
- No file → 400. Unknown result id → 404. Unhandled exception → friendly 500 page, with the stack trace only in the log.
- File names are sanitized with `werkzeug.utils.secure_filename` (truncated to 120 chars), and only for display and
  logging: `../../etc/passwd <script>.jpg` becomes `etc_passwd_script.jpg`. Jinja auto-escaping also blocks HTML injection.
- No image on disk: the upload is read into memory, the preview is an inline base64 thumbnail inside the HTML, and
  only metadata is stored. Werkzeug normally spools uploads larger than 500 KB to a temp file; the app overrides
  `Request._get_file_stream` to return an in-memory `BytesIO` (verified with a 7.7 MB upload).
- Friendly error pages: one `error.html` template for every HTTP error, with "try another image" / "history" links.

## 4.3 Frontend

Jinja templates extending `base.html` (top menu: Upload · History · Health; footer: current model and threshold).
- The upload page explains the model's limits before the user uploads.
- The result page shows the verdict (colour-coded), a confidence bar, P(AI), the threshold, the model version,
  latency and UTC time, plus an honest caveat. The caveat gets stronger when confidence is below 70 % and reminds
  users that "likely real" is not a guarantee.
- The history page shows the table described in Task 5.

## 4.4 Configuration and production server

- Every setting comes from environment variables (`web/config.py`): `MODEL_PATH`, `MODEL_CONFIG_PATH`,
  `THRESHOLD` (optional override), `DATABASE_URL`, `MAX_UPLOAD_MB`, `SECRET_KEY`, `HISTORY_PAGE_SIZE`,
  `LOG_LEVEL`. `.env.example` documents them. The same code runs locally (SQLite) and in Docker (PostgreSQL) with
  different env vars only.
- Served by gunicorn (`gunicorn -c gunicorn.conf.py "visionguard.web:create_app()"`): 1 worker × 4 threads.
  Each extra worker would load another TensorFlow copy, and TF does not survive `fork()`. The model call is
  serialized with a lock.
- Debug mode is off (`/health` reports `"debug": false`). Flask's debug mode enables the Werkzeug interactive
  debugger, which lets anyone who can trigger an error run arbitrary Python on the server. It also leaks stack
  traces and source code, and its auto-reloader re-imports the app (loading the model twice). The built-in
  development server is single-process and not hardened, which is why a production WSGI server is used.

## Task 5 — database trail

- SQLAlchemy 2 model `predictions`: `id`, `created_at` (UTC, indexed), `filename` (sanitized), `image_format`,
  `width`, `height`, `file_size`, `label` (indexed), `confidence`, `p_ai`, `threshold`, `model_name`,
  `model_version`, `latency_ms`, `source` (where inference ran: `local` now, `remote-api` in Task 8).
- That answers: what was decided (`label`, `p_ai`), when (`created_at`), by which model (`model_name` +
  `model_version` = registry version), and how confidently (`confidence` and the `threshold` in force at the time).
- One row is inserted inside the `/predict` view after every successful prediction. Rejected uploads are not
  stored; they are logged instead.
- `DATABASE_URL` selects the backend: SQLite for local runs, PostgreSQL in Docker Compose (Task 6). `pool_pre_ping`
  survives database restarts, and the schema is created at start-up.
- History page: newest first, 20 per page, counts per label, each id linking to `/result/<id>`.
- Privacy: no image bytes and no client IP are stored.
- Query example for the report (local SQLite):
  `sqlite3 instance/visionguard.db "select id, created_at, label, round(confidence,3), model_name, model_version from predictions order by id desc limit 10;"`
