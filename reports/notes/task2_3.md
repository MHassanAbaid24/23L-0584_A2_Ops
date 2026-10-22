# Tasks 2 & 3 — draft notes for the report (rewrite in your own words)

Runs were trained on a Kaggle GPU (`notebooks/kaggle_train.ipynb`) and logged to DagsHub MLflow, experiment `visionguard`.
Figures for every run are in `reports/figures/runs/<run>/` (curves, val/test confusion matrices, ROC, layer tables).
Comparison table: `reports/experiments/comparison.md`.

## 2.1 Input pipeline (`src/visionguard/data.py`)

- Stages: split CSV → (path, label) → `read_file` → `decode_image(channels=3)` (decodes by content, so the
  20 files with wrong extensions work) → resize to 224×224 → uint8 → `cache()` → shuffle (train only) → batch 32 →
  float32 → augment (train only) → `prefetch(AUTOTUNE)`.
- 224 px is the size the ImageNet backbones were pre-trained at; caching 974 images as uint8 takes ≈147 MB.
- Aspect ratio: `stretch` distorts it (median 1.5 → 1.0); `pad` letterboxes. E6 compares the two (below).
- Augmentation (train only): horizontal flip, ±18° rotation, ±10 % zoom, ±10 % contrast. Blur, noise and JPEG
  re-compression are avoided because they would destroy or imitate generation artefacts.
- Normalisation lives inside each model, so serving only decodes and resizes with the same `decode_and_resize`.
- Figure: `reports/figures/augmented_batch.png`.

## 2.2 Baseline CNN (`models.build_cnn`, run `cnn-baseline`)

- Rescaling(1/255) → 4 × [Conv3×3 (32/64/128/256) → BatchNorm → ReLU → MaxPool2] → GlobalAveragePooling →
  Dropout 0.3 → Dense(1, sigmoid). 390,113 parameters (table: `reports/figures/runs/cnn-baseline/layer_table.csv`).
- Loss: binary cross-entropy (sigmoid output is P(ai)). Optimizer: Adam 1e-3 (adaptive step sizes; a robust
  default). Metrics: accuracy, AUC, precision, recall (accuracy alone hides which error type occurs).
- Result: best epoch 11, val AUC 0.696, test AUC 0.742, test accuracy 0.687. With only 681 training images a
  from-scratch CNN cannot learn strong features, which motivates transfer learning.

## 2.3 Transfer learning (`models.build_transfer`, runs E0–E7)

- EfficientNetB0 / MobileNetV2 from `keras.applications`, `include_top=False`, ImageNet weights, GAP → Dropout →
  Dense(1, sigmoid).
- Preprocessing: EfficientNetB0 has its own rescaling/normalisation layers and expects raw [0, 255] input;
  MobileNetV2 expects [-1, 1], done with `Rescaling(1/127.5, offset=-1)` (identical to its `preprocess_input`).
  The pretrained filters only work on the input distribution they were trained on.
- Stage 1: backbone frozen, head trained at lr 1e-3 (up to 20 epochs). Stage 2: only the last block unfrozen
  (`block7a_*`/`top_conv` for EfficientNetB0, `block_16_*`/`Conv_1` for MobileNetV2), BatchNorm kept frozen,
  lr 1e-5 (up to 15 epochs). Trainable layer lists per stage: `trainable_layers.json` in each run folder.

## 2.4 Training discipline

- `keras.utils.set_random_seed(42)` + `enable_op_determinism()`; split seed 42.
- EarlyStopping(val_loss, patience 6, restore best), ModelCheckpoint(val_loss, best only; one instance across both
  stages so it keeps the global best), ReduceLROnPlateau(val_loss, ×0.3, patience 2).
- Best epochs: E0 35, E5 33, E7 24, CNN 11 (red dashed line in each `curves.png`).

## 2.5 Evaluation and threshold

- Threshold chosen on the validation set by maximising F0.5 for the "ai" class (precision weighted over recall);
  the test set is evaluated once per final model.
- Which error is worse? Calling a real photo fake. It is a false accusation against a person's genuine work and
  undermines trust; missing a fake is bad but the caveat on the result page already says the verdict is not proof.
  Hence F0.5 and a threshold above 0.5 where the data supports it.
- Champion E5 (MobileNetV2), threshold 0.6725: test accuracy 0.782, precision 0.902, recall 0.679, F1 0.775,
  ROC-AUC 0.871, real→fake rate 9.1 % (vs 18–35 % for the other models).
- CNN vs transfer (test): AUC 0.742 vs 0.871–0.892; accuracy 0.687 vs 0.769–0.830.

## 2.6 Error analysis (`reports/figures/champion/errors_grid.png`, `errors_table.csv`)

The 10 most confident mistakes of the champion (32 errors on 147 test images):
1. `RealArt/_MG_5812.jpg` (real → ai, p 0.98): close-up of a paint palette; paint-smear textures look like diffusion texture.
2. `240201-airec-main-ksp…` (ai → real, 0.06): flat pop-art/graphic style without photographic detail.
3. `midjourney-putin.jpg` (ai → real, 0.08): muted, collage-like Midjourney image that imitates a printed photo.
4. `RealArt/images108.jpg` (real → ai, 0.88): 300×168 thumbnail of a vividly painted portrait; saturated colours plus tiny size.
5. `…generative-ai-landscape-painting…` (ai → real, 0.14): carries a dreamstime watermark and stock-photo framing.
6. `Group_12467.jpg` (ai → real, 0.15): a logo-like line drawing; there is no texture for the model to use.
7. `3e9bc32a…1024x1024.jpg` (ai → real, 0.18): photoreal elderly portrait; modern generators match photo statistics.
8. `…yellow-kimono…800x800.jpg` (ai → real, 0.19): photoreal portrait with natural bokeh, 48 KB (heavily compressed).
9. `many-happy-diverse…260nw…` (real → ai, 0.78): grid of faces with a Shutterstock watermark; repeated faces look generated.
10. `chayka_ai_copyright.jpg` (ai → real, 0.23): a 3-D render; render-like real art also exists in the real class.

Pattern: the model has learned "painterly/saturated ⇒ AI" and "photo-like ⇒ real". It misses photoreal AI
portraits and flags real paintings or palettes. This is the dataset bias seen in the EDA.

## Bonus — Grad-CAM (`reports/figures/champion/gradcam.png`)

Heat maps over the last MobileNetV2 feature map. For correct AI predictions the evidence spreads over
high-saturation textured regions (swirls, faces in a collage). For the wrong ones it focuses on paint blobs (the
palette), a single earring or a colour patch, i.e. local texture and colour rather than global structure.

## 2.7 Keras Tuner

- Toy (Fashion-MNIST, Hyperband, `labs/keras_tuner_toy.py`): best units 224, lr 3e-4, val accuracy 0.849
  (`reports/tuner/toy_results.md`).
- VisionGuard (`src/visionguard/tune.py`): RandomSearch, 10 trials, EfficientNetB0, objective val AUC
  (`reports/tuner/visionguard_results.md`). Best trial: dropout 0.5, 80 unfrozen layers, fine-tune lr 3e-5 → val AUC 0.889.
- What mattered most: the number of unfrozen layers (mean val AUC 0.842 for 0 → 0.874 for 20 → 0.884 for 80;
  spread 0.042), then the fine-tune lr (3e-6 was clearly too small, 0.840; spread 0.034). Dropout mattered least
  (spread 0.010).
- Overfitting the validation set: with 146 validation images, AUC differences of ~0.005 are within noise
  (one image ≈ 0.7 % accuracy). Picking the max over many trials selects partly on luck, so the chosen score is
  optimistically biased. That is why the test set is kept separate and the final choice also weighs the
  real→fake rate and model size instead of the single highest number.

## 3.1 / 3.2 MLflow toy and DagsHub

- `labs/mlflow_toy.py`: MNIST dense net, two runs (lr 1e-3 → val acc 0.872, lr 1e-2 → 0.9105); params, per-epoch
  metrics via a Keras callback, `curves.png`, model. Local (`sqlite:///mlflow.db`) and DagsHub (experiment `toy-mnist`).
- Definitions (write your own): experiment = named group of runs for one question; run = one execution of
  training code; parameter = an input setting fixed for the run; metric = a numeric result, possibly per step;
  artifact = a file output (plot, model, CSV); tag = free-form label/metadata for searching and provenance.
- Credentials: `.env` (git-ignored) holds MLFLOW_TRACKING_URI/USERNAME/PASSWORD; `python-dotenv` loads it.

## 3.3 Manual logging vs autolog (E0-base vs E1-base-autolog, same configuration)

- Tags on both: `git_commit`, `git_dirty`, `dvc_data_md5` (8f23d6dc…, from `data/raw.dvc`), `dvc_data_nfiles`, `seed`.
- Manual (E0): 30 params describing the experiment (backbone, img_size, augment, resize_mode, fine_tune_lr,
  dataset sizes, TF/Keras versions, threshold), per-epoch metrics for all 35 epochs including the lr, test metrics.
- Autolog (E1): 31 params, but they describe `fit()` and the optimizer (opt_beta_1, opt_epsilon, shuffle,
  validation_steps…). It knows nothing about the data, backbone, augmentation or image size. It adds
  a TensorBoard log, model summary and checkpoints automatically. Missed: the second `fit()` call (fine-tuning)
  restarts its step counter, so the per-epoch history shows only the 20 stage-1 epochs; the threshold and test
  metrics only exist because the training code logs them explicitly.
- Conclusion: autolog is good for zero-effort optimizer/fit details; manual logging is needed for anything that
  defines the experiment and for multi-stage training.

## 3.4 Artifacts (every run)

`figures/` (curves, confusion matrices val/test, ROC), `splits/` (train/val/test CSVs), `run_info/`
(requirements.txt = pip freeze, metrics.json, serving_config.json, model summary, layer table, trainable layers,
history), `keras_model/model.keras`, plus the MLflow model `model`.

## 3.5 Comparison (validation metrics drive the choice; test shown for reporting)

| run | change vs E0 | val AUC | val F1 | val real→fake | test AUC | test F1 |
|---|---|---|---|---|---|---|
| E0-base | — | 0.870 | 0.789 | 0.169 | 0.885 | 0.815 |
| E1-base-autolog | logging only | 0.866 | 0.792 | 0.185 | 0.884 | 0.825 |
| E2-no-augment | no augmentation | 0.876 | 0.847 | 0.262 | 0.892 | 0.836 |
| E3-frozen-only | no fine-tuning | 0.857 | 0.832 | 0.308 | 0.874 | 0.805 |
| E4-img160 | 160 px input | 0.830 | 0.783 | 0.262 | 0.869 | 0.820 |
| E5-mobilenetv2 | MobileNetV2 backbone | **0.881** | 0.783 | **0.092** | 0.871 | 0.775 |
| E6-resize-pad | pad, not stretch | 0.859 | 0.797 | 0.215 | 0.888 | 0.850 |
| E7-ft-lr-1e-4 | fine-tune lr 1e-4 | 0.879 | 0.825 | 0.200 | 0.890 | 0.836 |
| cnn-baseline | from-scratch CNN | 0.696 | 0.671 | 0.308 | 0.742 | 0.693 |

Findings: fine-tuning helps (E3 < E0); smaller images hurt (E4); padding vs stretching makes little difference
(E6); a 10× larger fine-tune lr helps (E7); augmentation did not help on this small validation set (E2).

Choice: **E5-mobilenetv2** as champion. It has the highest validation AUC (threshold-free ranking quality) and, at
its validation-chosen threshold, by far the lowest rate of calling real images fake (9 %), which is the error that
matters most for this product. It is also the smallest model (16.7 MB .keras file, 2.3 M params), which suits a
CPU container. Trade-off: it misses more AI images (val fake→real 31 %). E7 is kept as challenger.
Caveat: test AUC ranks E5 below several EfficientNet runs; with 147 test images these differences (≈0.02) are
within noise, and choosing by test score would leak the test set into model selection.

## 3.6 Registry

Registered model `23L-0584_A2_Ops_visionguard`: v1 = E0-base, v2 = E7-ft-lr-1e-4 (alias `challenger`),
v3 = E5-mobilenetv2 (alias `champion`). Each version has tags val/test AUC, F1 and threshold.
`scripts/predict_champion.py <image> --tracking dagshub` loads `models:/23L-0584_A2_Ops_visionguard@champion`:
`RealArt/images37.jpg` → real (p_ai 0.022), `AiArtData/sddefault.jpg` → ai (p_ai 0.733, threshold 0.6725).
What went wrong: DagsHub could not resolve `runs:/<id>/model` for MLflow 3 LoggedModels; registering by
`models:/<model_id>` fixed it.

## 3.7 How the model reaches the server

`scripts/export_champion.py` resolves the `champion` alias once and writes `models/champion.keras` +
`models/serving_config.json` (threshold, image size, resize mode, version, run id). The app and the API load
these local files at start-up; they never call DagsHub per request. A new champion only requires re-running the
export and restarting the container.
