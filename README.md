# 23L-0584_A2_Ops — VisionGuard

VisionGuard is a web service that classifies an uploaded image as **AI-generated** or **real**.

- Model: TensorFlow/Keras, trained on [AI Generated Images vs Real Images](https://www.kaggle.com/datasets/cashbowman/ai-generated-images-vs-real-images)
- Experiment tracking and data/model versioning: MLflow + DVC on DagsHub
- Serving: Flask + database, packaged with Docker Compose; model API deployed to the cloud

## Setup

```bash
cp .env.example .env   # then fill in your DagsHub credentials
```
