# MLflow toy MNIST lab

Run the two small local MNIST experiments from the project root:

```bash
.venv/bin/python labs/mlflow_toy.py --target local
```

Open the local MLflow UI:

```bash
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open the URL printed by MLflow, select the two `toy-mnist` runs, then choose **Compare** to compare their parameters and metrics. The local SQLite database (`mlflow.db`) is generated output and should be git-ignored before committing coursework.

For the DagsHub variant, put `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` in the git-ignored project-root `.env` file, then run:

```bash
.venv/bin/python labs/mlflow_toy.py --target dagshub
```

The script prints the derived DagsHub experiments URL after both runs complete. It reads credentials only from `.env` and never prints them.
