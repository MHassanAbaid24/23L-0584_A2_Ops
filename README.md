# 23L-0584_A2_Ops — VisionGuard

VisionGuard is a web service that classifies an uploaded image as **AI-generated** or **real**.

- Model: TensorFlow/Keras, trained on [AI Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images)
- Experiment tracking and data/model versioning: MLflow + DVC on DagsHub
- Serving: Flask + database, packaged with Docker Compose; model API deployed to the cloud

## Setup

```bash
cp .env.example .env   # then fill in your DagsHub credentials
```

## Run the web app locally (Tasks 4–5)

```bash
python scripts/export_champion.py --tracking dagshub --out models   # once: fetch the champion model
gunicorn -c gunicorn.conf.py "visionguard.web:create_app()"        # http://localhost:8000
```

Pages: `/` upload · `/history` · `/result/<id>` · `/health`. Settings come from environment variables
(see `.env.example`). Debug mode is never enabled; the app always runs under gunicorn.
