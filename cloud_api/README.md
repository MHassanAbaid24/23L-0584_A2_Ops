# VisionGuard cloud API

This directory is the Vercel project root. It serves the float32 TFLite champion with `ai-edge-litert`.

## Endpoints

- `GET /api/health` is public and returns the deployed model metadata.
- `POST /api/predict` requires `X-API-Key` and multipart field `image`. It accepts JPEG, PNG, WEBP, GIF, and BMP images up to 4 MB.

```bash
curl https://23l-0584-a2-ops.vercel.app/api/health
curl -X POST https://23l-0584-a2-ops.vercel.app/api/predict \
  -H "X-API-Key: $MODEL_API_KEY" \
  -F "image=@data/raw/AiArtData/1000_F_551420122_DMVybsWtGw1kS6ogwlk66jcm78PmUIgn.jpg"
```

## Redeploy

Export the local environment without printing secrets, then deploy from this directory:

```bash
set -a; . ../.env; set +a
npx --yes vercel@latest deploy --prod --yes --token "$VERCEL_TOKEN"
```

`model/champion.tflite` and `model/serving_config.json` must be generated first with `../.venv/bin/python ../scripts/export_tflite.py`. Set the production `VG_API_KEY` in Vercel to the same value as local `MODEL_API_KEY` before the first deployment.
