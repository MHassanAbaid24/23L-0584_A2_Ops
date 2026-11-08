# Tasks 7 and 8 — facts, decisions and problems (source for the report)

Evidence: `reports/evidence/task7/*.txt` (+ `newman_results.json`, key redacted), `reports/evidence/task8/*.txt`;
screenshots `reports/screenshots/task7/`, `reports/screenshots/task8/` (transcripts rendered with
`scripts/render_transcripts.py`; browser captures of the running app / public URLs carry a context strip with URL, time,
user@host). Code: `scripts/export_tflite.py`, `cloud_api/` (api/index.py, requirements.txt, vercel.json, .vercelignore,
README.md, postman/VisionGuard.postman_collection.json), `src/visionguard/remote.py`, `src/visionguard/web/{config,__init__}.py`,
`tests/test_remote.py`, `compose.yaml` (INFERENCE_MODE / MODEL_API_URL / MODEL_API_KEY).

## Choice of platform (Task 7)
- Brief: put the model on a cloud/Vercel and call it from Flask; Flask itself runs on localhost.
- Constraints: no credit card. Oracle, AWS, Google Cloud need a card; Azure for Students was not available to me;
  Hugging Face Docker Spaces now need a paid plan (their docs, Nov 2026). Vercel Hobby is free, signs up with GitHub,
  no card → chosen (Vercel is named in the brief).
- Vercel limits that shaped the design: Python function bundle ≤ 500 MB, 2 GB RAM / 1 vCPU, 300 s max duration,
  4.5 MB request body, no persistent disk. TensorFlow (~1.2 GB installed) does not fit → convert the champion to TFLite
  (8,866,508 bytes) and run it with Google's `ai-edge-litert` 2.3.0 runtime (21.8 MB wheel). The database stays local
  (Compose PostgreSQL), so Vercel's lack of persistent disk does not matter: the API is stateless.
- Parity (scripts/export_tflite.py, 147 test images): Keras vs TFLite with identical input max |Δp| 3.2e-06;
  full pipeline TF-resize (training) vs Pillow-resize (API) mean |Δp| 0.0048690, max 0.0485432, 0 decision changes at
  threshold 0.6725, accuracy 0.7823 for both.

## API design
- `GET /api/health` (public, no secrets) → status, model_name 23L-0584_A2_Ops_visionguard, model_version 3,
  source_run E5-mobilenetv2, threshold 0.6725, img_size 224, runtime tflite.
- `POST /api/predict` multipart `image`, header `X-API-Key` (constant-time compare, key in Vercel env var VG_API_KEY,
  set with `vercel env add`; locally MODEL_API_KEY in the untracked .env) → label, confidence, p_ai, threshold,
  model_name/version, source_run, latency_ms, runtime. Errors as JSON: 401 missing/wrong key, 400 undecodable,
  415 unsupported (TIFF), 413 > 4 MB, 405 wrong method. Model + config loaded once per function instance (module import).
- Production URL: https://23l-0584-a2-ops.vercel.app (project `23l-0584-a2-ops`; Vercel lowercases names).
  Deployment id dpl_8PvQVg63S5ruSPWSBrk2RHSH4EKY (first full deployment); redeployed once after changing model_name.
- Latency: model inference ≈ 11–13 ms on Vercel; whole request ≈ 0.3–2.5 s from Pakistan (network + TLS + cold start).
  First request to a fresh deployment 2453.7 ms total / 12.99 ms inference; next one 1882.8 ms / 11.05 ms.

## Deployment steps (what was done, in order)
1. Account: vercel.com → Continue with GitHub (Hobby, no card); token created at vercel.com/account/tokens and stored
   as VERCEL_TOKEN in the untracked .env.
2. `python scripts/export_tflite.py` → cloud_api/model/champion.tflite + serving_config.json, prints the parity check.
3. Local test of the API (Flask dev server on 127.0.0.1:5057) with curl for every contract case (7_01).
4. `cd cloud_api && npx vercel link --yes --project 23l-0584-a2-ops --token …` → creates/links the project.
5. `vercel env add VG_API_KEY production` (value piped from .env, never typed on screen).
6. `vercel deploy --prod --yes` → builds the Python function from requirements.txt, uploads api/ + model/ (7_02).
7. Production was behind Vercel Authentication (Deployment Protection) → public requests got a login page.
   Disabled SSO protection for the project via the API (PATCH /v9/projects/{id} ssoProtection: null); the API stays
   protected by its own key.
8. Remote curl tests (7_03), Postman collection run with newman (7_04), browser check of /api/health (7_05).

## Problems (what went wrong)
- `tf.io.read_file` rejected a pathlib.Path in the exporter → pass str(path).
- First deployment shipped only two files: `.vercelignore` used directory negations that did not re-include the files
  inside → rewrote as `!api`, `!api/**`, `!model`, `!model/**`; the next deployment included the entrypoint and model.
- Local 401s during the first local test: the API reads VG_API_KEY while .env stores MODEL_API_KEY → map the variable
  when starting the local server.
- Deployment Protection blocked public access (step 7).
- The API first reported model_name = run name (E5-mobilenetv2) while the local app reports the registered model name;
  changed the API to return the registered name (source_run keeps the run name) and redeployed so the history page names
  the model consistently.

## Postman (Task 7)
Collection `cloud_api/postman/VisionGuard.postman_collection.json`: Health, Predict real image, Predict AI image,
Predict without key (401), Predict wrong key (401), Predict TIFF (415); variables baseUrl and apiKey (empty in the file,
passed at run time). Run with Postman's CLI runner newman against production: 6 requests, 12 assertions, 0 failed
(latest run 14.2 s, average response 2.3 s). The same collection can be imported into the Postman app (File → Import),
with apiKey set in the collection variables, to repeat the tests in the GUI.

## Task 8 — Flask ↔ API integration
- `INFERENCE_MODE=remote` (plus MODEL_API_URL, MODEL_API_KEY) makes create_app use `RemotePredictor` instead of loading
  the Keras model. It calls /api/health once at start-up (fail fast), then for each upload: validate locally (same
  checks as Task 4), preprocess with the training function `decode_and_resize` (TF, 224 px stretch), encode the 224×224
  uint8 array as PNG (lossless, ~100 KB ≪ 4.5 MB limit), POST to /api/predict with the key, store the result with
  source = "vercel" and the API's model name/version. Because the API receives an image that is already 224×224 it does
  not resize again → the Flask path uses exactly the training preprocessing. Errors: rejected key → 502 page, API
  down/timeout → 503 page; no row is stored on failure. Only the standard library (urllib) is used for HTTP.
- Tests: tests/test_remote.py runs a local stand-in API on a random port: rows stored with source vercel, the API receives
  exactly img_size×img_size PNG, wrong key → error page and no row, API down → 503 and no row, unreachable health at
  start-up fails fast. Total suite 18 tests passing.
- Demo (Compose stack, web rebuilt with the Task 8 code, `INFERENCE_MODE=remote docker compose up -d --no-deps web`):
  /health shows inference_mode remote, api_health true, model v3, runtime tflite. Six images uploaded through the browser
  UI: whats-world-population (real 99.6 %), midjourney20images (AI 99.8 %), cheetah-003 (real 80.1 %), images49 (AI 95.1 %),
  mouse-in-a-maze (AI 68.5 %, borderline), midjourney-putin (real 92.4 % — a missed fake). The verdicts and confidences
  are identical to the local model's results for the same files in Chapter 4, which confirms that preprocessing matches.
- Latency: 792–1804 ms per remote prediction (avg 1133 ms over 6) vs 391 ms for the one local prediction in the same DB;
  the very first remote request took 5490 ms end to end (first TF call in the web process + first TLS connection).
- History page (from PostgreSQL) shows SOURCE vercel for rows 2–7 and local for row 1; psql query confirms it (8_02).
- Build note: `docker compose build` uses BuildKit, whose cache is separate from the legacy `docker build` used earlier,
  so its first build reinstalled the dependencies (≈273 s); the second build took 2.7 s with every layer CACHED.
