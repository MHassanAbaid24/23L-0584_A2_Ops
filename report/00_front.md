# VisionGuard: AI-Generated vs Real Image Classification and Deployment

MLOps — Assignment 2 · Project `23L-0584_A2_Ops`

| | |
|---|---|
| Name | Muhammad Hassan Abaid |
| Roll number | 23L-0584 |
| GitHub repository | <https://github.com/MHassanAbaid24/23L-0584_A2_Ops> |
| DagsHub project (code, data, experiments, model registry) | <https://dagshub.com/MHassanAbaid24/23L-0584_A2_Ops> |
| DVC toy-lab repository (Task 1.4) | <https://dagshub.com/MHassanAbaid24/23L-0584_A2_Ops_dvc_toy> |
| Model API URL (Task 7) | {{API_URL}} |
| Live application URL | {{LIVE_URL}} |
| Commit hash of the graded version | `{{COMMIT}}` |
| Report built | {{BUILD_DATE}} |

## Disclosure of AI assistants and tutorials

**AI assistants.** I used an AI coding assistant throughout this assignment: Anthropic's Claude, running in the *omp*
coding-agent command-line tool on my machine. Some well-specified sub-tasks were delegated by that tool to OpenAI's
GPT-5.6 model: the MLflow and Keras Tuner toy-lab scripts, the model-registry scripts and the Kaggle notebook.
I used the assistant for:

- breaking the brief into tasks and a dependency plan;
- writing most of the code: the data audit, the pipeline, the models, training, tracking, the Flask application,
  and the scripts and labs in the repository;
- running local smoke tests and verification commands;
- capturing the Task 4 and Task 5 screenshots from the running application with a headless browser
  (these images carry an added context strip with URL, time, user@host and project name);
- drafting the text of this report from my notes, the logged metrics and the generated figures.

**My own work.**

- I created and configured the GitHub, DagsHub and Kaggle accounts, and supplied credentials only through git-ignored
  environment files and Kaggle Secrets.
- I ran the GPU training, experiment and tuning jobs in a Kaggle notebook.
- I took the Task 1–3 screenshots on my own desktop.
- I made or approved the design decisions described in each chapter: for example, the choice of a free cloud host
  without a credit card, and Kaggle GPUs for training.

**Documentation and tutorials consulted.** The official documentation of TensorFlow/Keras (tf.data, keras.applications,
Keras Tuner), MLflow (tracking, autologging, model registry), DVC, DagsHub, Flask, SQLAlchemy and gunicorn. No other
person's code or report was used.
