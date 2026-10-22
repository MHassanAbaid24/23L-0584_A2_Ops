"""Task 2.1 — tf.data input pipeline, shared by training and serving.

Stages (train):  split CSV → (path, label) tensors → read bytes → decode by content → RGB → resize
                 → uint8 → cache → shuffle → augment → float32 → batch → prefetch
Stages (val/test): same, minus shuffle and augment.

Design decisions
- `decode_and_resize` is the ONLY image-to-tensor code path. Serving (Task 4.1) imports it, so uploaded
  images get exactly the same treatment as training images.
- Decoding is content-based (`tf.io.decode_image` sniffs magic bytes): 20 files have misleading extensions.
- `channels=3` turns greyscale, palette and RGBA images into RGB; alpha is dropped.
- Pixel values stay in [0, 255]. Each model normalises inside its own graph (Rescaling for the CNN,
  the backbone's own `preprocess_input` for transfer learning). Serving therefore never needs to know
  which normalisation a model expects.
- Resizing: `stretch` (default) resizes straight to size×size and distorts the aspect ratio (median 1.5 → 1.0).
  `pad` keeps the aspect ratio and letterboxes with black. Both are exposed so their effect can be
  compared as an experiment (Task 3.5).
- Images are cached as uint8 after resizing: 974 × 224 × 224 × 3 B ≈ 147 MB instead of ~590 MB as float32,
  and the expensive JPEG/PNG decode happens once instead of every epoch.
- Augmentation runs after the cache (otherwise every epoch would see the same augmented copy) and only
  on the training split. Only geometry and mild contrast changes are used: no blur, noise or JPEG
  re-compression, because those would destroy or imitate the generation artefacts the model must learn.
"""

from __future__ import annotations

from pathlib import Path

import keras
import pandas as pd
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
DEFAULT_IMG_SIZE = 224
DEFAULT_DATA_ROOT = Path("data/raw")
DEFAULT_SPLITS_DIR = Path("splits")


def decode_and_resize(image_bytes: tf.Tensor, img_size: int = DEFAULT_IMG_SIZE,
                      resize_mode: str = "stretch") -> tf.Tensor:
    """Raw encoded bytes → uint8 tensor (img_size, img_size, 3). Used by training and serving."""
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    if resize_mode == "stretch":
        img = tf.image.resize(img, (img_size, img_size), method="bilinear", antialias=True)
    elif resize_mode == "pad":
        img = tf.image.resize_with_pad(img, img_size, img_size, method="bilinear", antialias=True)
    else:
        raise ValueError(f"unknown resize_mode {resize_mode!r}")
    return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)


def build_augmenter(seed: int | None = None) -> keras.Sequential:
    """Train-only random augmentation (Keras preprocessing layers, run inside tf.data)."""
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomRotation(0.05, fill_mode="reflect", seed=seed),   # ±18°
        keras.layers.RandomZoom((-0.1, 0.1), fill_mode="reflect", seed=seed),
        keras.layers.RandomContrast(0.1, seed=seed),
    ], name="augment")


def read_split(split: str, splits_dir: Path = DEFAULT_SPLITS_DIR) -> pd.DataFrame:
    return pd.read_csv(Path(splits_dir) / f"{split}.csv")


def make_dataset(split: str, *, img_size: int = DEFAULT_IMG_SIZE, batch_size: int = 32,
                 augment: bool | None = None, resize_mode: str = "stretch", seed: int = 42,
                 data_root: Path = DEFAULT_DATA_ROOT, splits_dir: Path = DEFAULT_SPLITS_DIR,
                 cache: bool = True) -> tf.data.Dataset:
    """Batched dataset of (float32 images in [0,255], float32 labels) for one split.

    `augment` defaults to True for the train split only.
    """
    is_train = split == "train"
    augment = is_train if augment is None else augment
    df = read_split(split, splits_dir)
    paths = [str(Path(data_root) / p) for p in df["path"]]
    labels = df["label"].astype("float32").to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(lambda p, y: (decode_and_resize(tf.io.read_file(p), img_size, resize_mode), y),
                num_parallel_calls=AUTOTUNE, deterministic=True)
    if cache:
        ds = ds.cache()
    if is_train:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
    if augment:
        augmenter = build_augmenter(seed)
        ds = ds.map(lambda x, y: (tf.clip_by_value(augmenter(x, training=True), 0.0, 255.0), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)
