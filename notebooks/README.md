# Kaggle training notebook

1. Upload `kaggle_train.ipynb` to Kaggle.
2. Attach the dataset `cashbowman/ai-generated-images-vs-real-images`.
3. Add Kaggle Secrets named `DAGSHUB_USER` and `DAGSHUB_TOKEN`.
4. In Notebook settings, select a **T4 or P100 GPU** and turn **Internet** on.
5. Use **Save Version → Save & Run All (Commit)** so Kaggle runs the notebook in the background.

The notebook clones the DagsHub mirror, verifies the attached images against the repository split hashes, runs the CNN baseline, the eight transfer-learning experiments, and the tuner. Expect roughly **3–6 hours** on a T4/P100, depending on early stopping and network speed.

Per-run figures, metrics, and models are written under `outputs/`; tuner reports are under `reports/tuner/`. The notebook also writes `/kaggle/working/experiment_comparison.csv` and `/kaggle/working/vg_outputs.zip`, both available from Kaggle's Output tab after the committed run finishes.
