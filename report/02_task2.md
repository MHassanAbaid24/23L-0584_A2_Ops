# Chapter 2 — Model Development with TensorFlow/Keras (14 marks)

I trained the models on a Kaggle GPU through `notebooks/kaggle_train.ipynb`. It clones the project from DagsHub, checks split-image SHA-256 hashes, then calls the same local entry points: `python -m visionguard.train …` and `python -m visionguard.tune …`. Chapter 1 supplies 681 training images (305 real, 376 AI), 146 validation images (65 real, 81 AI), and 147 test images (66 real, 81 AI). Label 1 is "AI-generated" throughout.

## 2.1 Input pipeline

`make_dataset(split)` in `src/visionguard/data.py` reads `splits/train.csv`, `val.csv`, or `test.csv` and builds a `tf.data.Dataset`:

1. **Paths and labels.** `from_tensor_slices` turns each relative path and 0/1 label into tensors.
2. **Read and decode.** `tf.io.read_file` and `tf.io.decode_image(channels=3)` read by file content, not extension. This handles the 20 audit findings with mismatched extensions and converts greyscale, palette, and RGBA images to RGB.
3. **Resize.** Images become 224×224 with bilinear, anti-aliased resizing. Both ImageNet backbones were trained at 224 px. I kept the simpler `stretch`: it changes the median 1.5:1 image to 1:1, but E6 found validation AUC 0.8587 with padding and 0.8697 with stretching (Chapter 3), which is within noise.
4. **Cache.** Resized images are rounded to `uint8` and cached. For 974 images, that is about 147 MB rather than roughly four times as much for float32, and decoding happens once.
5. **Shuffle and batch.** Training uses a full shuffle buffer with seed 42 and batches of 32. Batches are cast to float32; models normalise internally (Sections 2.2 and 2.3).
6. **Augment and prefetch.** Training batches get horizontal flips, `RandomRotation(0.05)` (±18°, 5 % of a turn), zoom ±10 %, and contrast ±10 %. Reflection fills corners and output is clipped to [0, 255]. Augmentation follows the cache so every epoch differs. `AUTOTUNE` prefetches the next batch.

Validation and test data skip shuffling and augmentation. I left out blur, noise, and JPEG re-compression because they could remove or imitate the artefacts under study. `decode_and_resize` is also used by the Chapter 4 web app.

Figure 2.1 compares the resized batch with its augmented version. `scripts/show_augmented_batch.py` uses the unshuffled validation split so both rows contain the same images.

![Figure 2.1 — Augmented training batch at 224×224.](../reports/figures/augmented_batch.png)

## 2.2 Baseline CNN

`build_cnn` in `src/visionguard/models.py` created the `cnn-baseline` run. It has four 3×3 convolution blocks with 32, 64, 128, and 256 filters, each followed by BatchNormalization, ReLU, and 2×2 max pooling. The convolutions have no bias because BatchNormalization supplies the shift; for example, `conv1` has 3·3·3·32 = 864 weights. `Rescaling(1/255)` inside the model accepts raw pixels, while global average pooling avoids flattening the 14×14×256 output into 50,176 features for only 681 training images. Dropout 0.3 regularises the 256 pooled features, and `p_ai` is a one-unit sigmoid.

Table 2.1 gives the generated layer details. Figure 2.2 and Figure 2.3 show Keras' terminal summary: 390,113 parameters, of which 389,153 are trainable.

**Table 2.1 — Baseline CNN: output shape and parameters per layer (from `reports/figures/runs/cnn-baseline/layer_table.csv`).**

| Layer | Type | Output shape | Parameters |
|---|---|---|---:|
| image | InputLayer | (None, 224, 224, 3) | 0 |
| rescale_0_1 | Rescaling | (None, 224, 224, 3) | 0 |
| conv1 | Conv2D | (None, 224, 224, 32) | 864 |
| bn1 | BatchNormalization | (None, 224, 224, 32) | 128 |
| relu1 | Activation | (None, 224, 224, 32) | 0 |
| pool1 | MaxPooling2D | (None, 112, 112, 32) | 0 |
| conv2 | Conv2D | (None, 112, 112, 64) | 18,432 |
| bn2 | BatchNormalization | (None, 112, 112, 64) | 256 |
| relu2 | Activation | (None, 112, 112, 64) | 0 |
| pool2 | MaxPooling2D | (None, 56, 56, 64) | 0 |
| conv3 | Conv2D | (None, 56, 56, 128) | 73,728 |
| bn3 | BatchNormalization | (None, 56, 56, 128) | 512 |
| relu3 | Activation | (None, 56, 56, 128) | 0 |
| pool3 | MaxPooling2D | (None, 28, 28, 128) | 0 |
| conv4 | Conv2D | (None, 28, 28, 256) | 294,912 |
| bn4 | BatchNormalization | (None, 28, 28, 256) | 1,024 |
| relu4 | Activation | (None, 28, 28, 256) | 0 |
| pool4 | MaxPooling2D | (None, 14, 14, 256) | 0 |
| gap | GlobalAveragePooling2D | (None, 256) | 0 |
| dropout | Dropout | (None, 256) | 0 |
| p_ai | Dense | (None, 1) | 257 |
| **Total** | | | **390,113** |

![Figure 2.2 — Baseline CNN summary, part 1.](../reports/screenshots/task2/2.2_cnn_model_summary_1.png)

![Figure 2.3 — Baseline CNN summary, part 2.](../reports/screenshots/task2/2.2_cnn_model_summary_2.png)

`compile_model` in `src/visionguard/train.py` uses binary cross-entropy and Adam at 10⁻³, tracking accuracy, AUC, precision, and recall. Cross-entropy penalises confident mistakes; AUC measures ranking without choosing a threshold.

The CNN had up to 40 epochs and patience 8, but stopped after 19. Its best epoch was 11, with validation AUC 0.6957 and test AUC 0.7421 (Section 2.5). Starting from random weights is a poor fit for 681 training images, so I moved to transfer learning.

## 2.3 Transfer learning

`build_transfer` loads ImageNet weights from `keras.applications` with `include_top=False`, then adds GlobalAveragePooling, Dropout 0.3, and Dense(1, sigmoid). I used EfficientNetB0 for E0 and MobileNetV2 for E5, the Chapter 3 champion.

**Preprocessing.** EfficientNetB0 already contains Rescaling and Normalization and takes [0, 255] input. MobileNetV2 takes [−1, 1], so I used `Rescaling(1/127.5, offset=−1)` as `mobilenetv2_preprocess`, exactly matching x / 127.5 − 1. Saving a real layer rather than a `Lambda` also keeps the `.keras` file compatible with safe mode. The server therefore only decodes and resizes.

**Two training stages** in `run()`:

- **Stage 1 — frozen backbone.** `base.trainable = False`; only the 1,281-parameter head (1,280 weights + 1 bias) trains, at 10⁻³ for at most 20 epochs. This protects pretrained features while the random head settles.
- **Stage 2 — fine-tune the last block.** `unfreeze_top()` changes only the last block, then recompiles and trains for at most 15 more epochs at 10⁻⁵, a hundred times smaller. Earlier layers stay useful for general edges and texture; BatchNorm stays frozen and the backbone is called with `training=False` so batches of 32 do not replace ImageNet statistics.

Table 2.2 lists the recorded trainable layers. Stage 2 has 880,321 trainable parameters for MobileNetV2 and 1,122,865 for EfficientNetB0.

**Table 2.2 — Trainable layers in each stage (from `trainable_layers.json`).**

| Stage | E0 — EfficientNetB0 | E5 — MobileNetV2 (champion) |
|---|---|---|
| 1 (frozen backbone) | `p_ai` (head only) | `p_ai` (head only) |
| 2 (fine-tuning) | `block7a_expand_conv`, `block7a_dwconv`, `block7a_se_reduce`, `block7a_se_expand`, `block7a_project_conv`, `top_conv`, `p_ai` | `block_16_expand`, `block_16_depthwise`, `block_16_project`, `Conv_1`, `p_ai` |

One `curves.png` covers both `fit()` calls; stage 2 continues from `initial_epoch=20`. The grey dotted line marks epoch 20.5, and the red dashed line marks the best epoch.

Figure 2.4 shows E0 improving after unfreezing, while Figure 2.5 shows E5's validation loss drop again in stage 2. E3, with a frozen backbone only, reached validation AUC 0.8572 against 0.8697 for E0, so fine-tuning helped.

![Figure 2.4 — E0 training curves.](../reports/figures/runs/E0-base/curves.png)

![Figure 2.5 — E5 training curves.](../reports/figures/runs/E5-mobilenetv2/curves.png)

### What went wrong

E0's best epoch was the last, 35 of 35, while validation loss was still falling. Its 15-epoch fine-tuning budget may have been too short.

## 2.4 Training discipline

- **Fixed seeds.** `keras.utils.set_random_seed(42)` covers Python, NumPy, and TensorFlow; deterministic GPU kernels are enabled. Split, shuffle, and augmentation seeds are also 42.
- **Early stopping.** I monitor validation loss with patience 6, or 8 for the CNN, and `restore_best_weights=True`. It is less jumpy than accuracy on 146 images.
- **Checkpointing.** One `ModelCheckpoint(save_best_only=True)` instance spans both stages, then its checkpoint is reloaded. A new stage-2 callback would forget a better stage-1 epoch.
- **Rate reduction.** `ReduceLROnPlateau` multiplies the rate by 0.3 after 2 unimproved epochs, down to 10⁻⁷.

The lowest validation loss over the combined history is the best epoch. Table 2.3 records it, and Figure 2.6 shows the CNN overfitting after epoch 11; training ends 8 epochs later with that checkpoint restored.

**Table 2.3 — Best epoch and epochs actually run (from each run's `metrics.json`).**

| Run | Best epoch | Epochs run | How training ended |
|---|---:|---:|---|
| cnn-baseline | 11 | 19 | early stopping (11 + patience 8) |
| E0-base | 35 | 35 | epoch budget used up (20 + 15) |
| E5-mobilenetv2 (champion) | 33 | 35 | epoch budget used up |
| E7-ft-lr-1e-4 | 24 | 30 | early stopping in stage 2 (24 + patience 6) |

![Figure 2.6 — Baseline CNN training curves.](../reports/figures/runs/cnn-baseline/curves.png)

## 2.5 Evaluation

**Threshold on validation.** `choose_threshold()` in `src/visionguard/evaluate.py` tries every distinct validation probability and 0.5, then keeps the threshold with the highest AI-class F0.5 score. F-beta is (1 + β²)·P·R / (β²·P + R); β = 0.5 gives precision more weight, and ties go to the higher threshold. The selected value is saved and used unchanged for test and the web app. For E5, Youden's J would instead be 0.6208.

Calling a real image fake is worse than missing a fake: it can accuse someone of faking genuine work. I therefore prioritised AI precision. On E5 validation, 0.5 gives accuracy 0.8151 and F1 0.8364; at 0.6725, accuracy is 0.7877 and F1 0.7832, but precision reaches 0.9032 and only 6 of 65 real images (9.23 %) are called fake.

The test set is used once, after checkpoint and threshold selection. Chapter 3 chooses the champion from validation metrics. Table 2.4 compares the CNN, E5, and E0.

**Table 2.4 — Test-set metrics at each model's validation-chosen threshold (147 images; values to 4 decimals as in `reports/experiments/comparison.md`, rates from `metrics.json`).**

| Metric | CNN baseline | E5-mobilenetv2 (champion) | E0-base (EfficientNetB0) |
|---|---:|---:|---:|
| Threshold (from validation) | 0.5422 | 0.6725 | 0.6108 |
| Accuracy | 0.6871 | 0.7823 | 0.8027 |
| Precision (AI) | 0.7536 | 0.9016 | 0.8421 |
| Recall (AI) | 0.6420 | 0.6790 | 0.7901 |
| F1 | 0.6933 | 0.7746 | 0.8153 |
| ROC-AUC | 0.7421 | 0.8709 | 0.8853 |
| Real called fake (FP rate) | 0.2576 | **0.0909** | 0.1818 |
| Fake called real (miss rate) | 0.3580 | 0.3210 | 0.2099 |
| TN / FP / FN / TP | 49 / 17 / 29 / 52 | 60 / 6 / 26 / 55 | 54 / 12 / 17 / 64 |

Transfer learning beats the CNN. E0 has higher test accuracy, F1, and AUC than E5, but its real-called-fake rate is 0.1818 against 0.0909. E5 won on validation (Chapter 3); an AUC gap of about 0.015 on 147 images is within noise, and choosing on test performance would leak the test set.

Figure 2.7 and Figure 2.8 show 17 false accusations for the CNN and 6 for E5, though E5 misses 26 AI images. Figure 2.9 and Figure 2.10 show E5's validation and test curves close together at AUC 0.881 and 0.871; the CNN is closer to the diagonal.

![Figure 2.7 — Baseline CNN test confusion matrix.](../reports/figures/runs/cnn-baseline/confusion_matrix_test.png)

![Figure 2.8 — Champion E5 test confusion matrix.](../reports/figures/runs/E5-mobilenetv2/confusion_matrix_test.png)

![Figure 2.9 — Baseline CNN ROC curves.](../reports/figures/runs/cnn-baseline/roc_curve.png)

![Figure 2.10 — Champion E5 ROC curves.](../reports/figures/runs/E5-mobilenetv2/roc_curve.png)

## 2.6 Error analysis

The champion misclassifies 32 of 147 test images: 6 false positives and 26 false negatives. `scripts/analyze_errors.py` ranks them by confidence and joins their Chapter 1 audit facts. Figure 2.11 shows the 10 most confident mistakes after stretch resizing; Table 2.5 records my explanations.

![Figure 2.11 — Champion E5's 10 most confident test errors.](../reports/figures/champion/errors_grid.png)

**Table 2.5 — Hypotheses for the 10 most confident test errors (facts from `reports/figures/champion/errors_table.csv`).**

| # | File | True → predicted | p(ai) | Size, file size | Hypothesis |
|---:|---|---|---:|---|---|
| 1 | `RealArt/_MG_5812.jpg` | real → ai | 0.98 | 2500×1667, 870 KB | Saturated paint blobs resemble AI painterly texture. |
| 2 | `AiArtData/240201-airec-main-ksp-5625e8.jpg` | ai → real | 0.06 | 760×380, 180 KB | Flat pop art has little texture and is squeezed from 2:1. |
| 3 | `AiArtData/midjourney-putin.jpg` | ai → real | 0.08 | 1024×576, 376 KB | Muted paper-grain collage resembles a printed photo. |
| 4 | `RealArt/images108.jpg` | real → ai | 0.88 | 300×168, 12 KB | Upscaled saturated thumbnail resembles a smooth AI render. |
| 5 | `AiArtData/ath-flowers-vivid-colors-generative-ai-landscape-painting-…-281590810.jpg` | ai → real | 0.14 | 897×1690, 250 KB | Watermark and tall stretch resemble real stock art. |
| 6 | `AiArtData/Group_12467.jpg` | ai → real | 0.15 | 1548×776, 91 KB | Flat black-and-white line drawing offers little texture. |
| 7 | `AiArtData/3e9bc32a-…-1024x1024.jpg` | ai → real | 0.18 | 1024×1024, 112 KB | Photoreal portrait resembles the learned real class. |
| 8 | `AiArtData/earing-a-yellow-kimono-…-800x800.jpg` | ai → real | 0.19 | 800×800, 48 KB | Compression and resizing hide its small generator trace. |
| 9 | `RealArt/many-happy-diverse-ethnicity-different-260nw-1733872811.jpg` | real → ai | 0.78 | 520×280, 60 KB | Repeated faces resemble the AI collage in Figure 2.12. |
| 10 | `AiArtData/chayka_ai_copyright.jpg` | ai → real | 0.23 | 1500×1500, 638 KB | Render-like art appears in both classes. |

**Pattern.** Seven of the ten are AI images called real, which fits the Section 2.5 threshold. The model seems to use "painterly or saturated ⇒ AI" and "photo-like ⇒ real", missing photoreal AI and flat graphics while flagging real paint. More photoreal AI and real paintings in training would reduce that Chapter 1 class bias.

## 2.7 Keras Tuner

**Toy lab.** `labs/keras_tuner_toy.py` tunes a Fashion-MNIST classifier: Flatten → Dense(*units*, ReLU) → Dropout 0.2 → Dense(10, softmax). It searches units from 32 to 256 in steps of 32 and learning rate {10⁻², 3·10⁻³, 10⁻³, 3·10⁻⁴, 10⁻⁴} on 12,000 training and 3,000 validation images. Hyperband uses `max_epochs=6` and `factor=3`, cheaply dropping weak settings. In bracket 1, trials 0000–0004 ran 2 epochs and the best continued as 0005 and 0006; bracket 0 ran 0007–0009 for 6 epochs. The best was 224 units at 3·10⁻⁴, with validation accuracy 0.849333. Retraining for only 3 epochs reached 0.833667.

**Table 2.6 — Keras Tuner toy results, Fashion-MNIST, Hyperband (`reports/tuner/toy_results.md`).**

| Trial | Units | Learning rate | Epochs trained | Best val accuracy |
|---|---:|---:|---:|---:|
| 0005 | 224 | 0.0003 | 6 | 0.849333 |
| 0009 | 128 | 0.001 | 6 | 0.842333 |
| 0008 | 192 | 0.01 | 6 | 0.827667 |
| 0006 | 160 | 0.01 | 6 | 0.823667 |
| 0000 | 224 | 0.0003 | 2 | 0.822333 |
| 0004 | 160 | 0.01 | 2 | 0.820667 |
| 0003 | 224 | 0.003 | 2 | 0.820333 |
| 0002 | 160 | 0.001 | 2 | 0.818000 |
| 0001 | 96 | 0.0003 | 2 | 0.811667 |
| 0007 | 32 | 0.0001 | 6 | 0.784333 |

**Applied to VisionGuard.** `TwoStageTuner` in `src/visionguard/tune.py` extends RandomSearch with the Section 2.3 schedule and EfficientNetB0: up to 8 head-training epochs at 10⁻³, then up to 10 epochs after unfreezing the last *N* layers. Both stages use patience 3. It searches fine-tune learning rate 10⁻⁴, 3·10⁻⁵, 10⁻⁵, or 3·10⁻⁶; dropout 0.1, 0.3, or 0.5; and 0, 10, 20, 40, or 80 unfrozen layers. BatchNorm remains frozen.

The objective is maximum validation AUC at the lowest-loss epoch, so no threshold is needed and the test split remains unloaded. I ran 10 Kaggle trials with seed 42, covering 10 of 60 combinations; none sampled 10 or 40 unfrozen layers. Table 2.7 gives the results.

**Table 2.7 — VisionGuard tuner results (EfficientNetB0, RandomSearch, 10 trials, objective max val AUC; `reports/tuner/visionguard_results.md`).**

| Trial | Dropout | Unfrozen layers | Fine-tune LR | Val AUC | Val accuracy | Best epoch |
|---|---:|---:|---:|---:|---:|---:|
| trial-09 | 0.5 | 80 | 3e-05 | 0.8888 | 0.8219 | 17 |
| trial-05 | 0.1 | 20 | 0.0001 | 0.8844 | 0.8425 | 13 |
| trial-02 | 0.1 | 80 | 3e-05 | 0.8828 | 0.8151 | 13 |
| trial-07 | 0.3 | 20 | 3e-05 | 0.8799 | 0.8082 | 18 |
| trial-03 | 0.1 | 80 | 1e-05 | 0.8795 | 0.8082 | 18 |
| trial-06 | 0.1 | 20 | 3e-06 | 0.8561 | 0.7808 | 18 |
| trial-00 | 0.1 | 0 | 0.0001 | 0.8522 | 0.7671 | 18 |
| trial-01 | 0.3 | 0 | 3e-05 | 0.8461 | 0.7671 | 18 |
| trial-08 | 0.1 | 0 | 1e-05 | 0.8456 | 0.7808 | 18 |
| trial-04 | 0.5 | 0 | 3e-06 | 0.8246 | 0.7534 | 18 |

**Which setting mattered most?** Table 2.8 averages validation AUC by value. Unfrozen layers has the largest spread, 0.0416: ImageNet object features need adapting to generation artefacts. The learning-rate spread is 0.034, partly because 3·10⁻⁶ barely moves weights in 10 epochs. Dropout's spread is 0.0101. These are rough estimates from 2–4 trials per mean.

**Table 2.8 — Mean validation AUC per hyperparameter value (n = number of trials).**

| Hyperparameter | Mean val AUC per value | Spread |
|---|---|---:|
| Unfrozen layers | 0 → 0.8421 (n=4), 20 → 0.8735 (n=3), 80 → 0.8837 (n=3) | **0.0416** |
| Fine-tune LR | 3e-06 → 0.8404 (n=2), 1e-05 → 0.8625 (n=2), 0.0001 → 0.8683 (n=2), 3e-05 → 0.8744 (n=4) | 0.034 |
| Dropout | 0.5 → 0.8567 (n=2), 0.3 → 0.863 (n=2), 0.1 → 0.8668 (n=6) | 0.0101 |

Choosing the highest validation result can overfit validation. One of 146 images changes accuracy by about 0.7 %, and the top five trials are within 0.0093 AUC. The same validation set guides early stopping and threshold choice, so Chapter 3 also considers the real-called-fake rate and model size.

### What went wrong

Random search missed 10 and 40 unfrozen layers, so the 20-to-80 trend has only two points. I did not run the best tuner configuration, 80 layers, as a separate full experiment; Chapter 3 therefore uses the Section 2.3 last-block run.

## Bonus — Grad-CAM

`gradcam()` in `scripts/analyze_errors.py` records MobileNetV2's final 7×7×1280 feature map, averages the output p(ai) gradients per channel, weights the map, applies ReLU, and upsamples it to 224×224. It selects 2 correct images per class with seed 42 and the 4 most confident mistakes (#1–#4 in Table 2.5). Figure 2.12 shows the overlays.

![Figure 2.12 — Grad-CAM for champion E5.](../reports/figures/champion/gradcam.png)

Correct AI images (p = 0.98 and 0.99) activate textured, high-saturation areas; correct real images have p = 0.26 and 0.36. The wrong cases focus on local colour and texture, supporting the Section 2.6 shortcut rather than a global reading of the image. The 7×7 map is coarse, and each map is scaled to its own maximum, so red is only the strongest evidence within that image.
