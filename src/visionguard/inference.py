"""Task 4.1 — inference module: load the champion model ONCE, validate uploads, predict.

    python -m visionguard.inference data/raw/RealArt/images37.jpg data/raw/AiArtData/sddefault.jpg

Why load once: reading and building the Keras graph takes ~10 s (measured), a prediction ~0.1–0.7 s. Loading per
request would make every request 10+ s slower, multiply memory use under concurrency and waste CPU.

Why the preprocessing must match training: the model only learned the input distribution it saw. Serving uses the
very same `visionguard.data.decode_and_resize` (decode by content → RGB → antialiased resize → uint8 → float32 in
[0, 255]; normalisation happens inside the model). A different resize method, channel order or value range would
silently shift the predictions (train/serve skew) without raising any error.
"""

from __future__ import annotations

import io
import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

# Formats accepted from users. JPEG/PNG/GIF/BMP are decoded directly by TensorFlow, exactly as in training;
# WEBP is not supported by tf.io.decode_image, so it is converted losslessly to PNG first (identical pixels).
ALLOWED_FORMATS = {"JPEG", "PNG", "GIF", "BMP", "WEBP"}
TF_NATIVE_FORMATS = {"JPEG", "PNG", "GIF", "BMP"}
MAX_PIXELS = 100_000_000  # largest training image is 60 MP; anything larger is rejected (decompression bombs)


class InvalidImageError(ValueError):
    """The upload is not an image we can safely decode."""


@dataclass(frozen=True)
class ImageInfo:
    format: str
    width: int
    height: int
    mode: str
    size_bytes: int


@dataclass(frozen=True)
class Prediction:
    label: str            # "ai" or "real"
    confidence: float     # probability of the predicted label
    p_ai: float           # model output: probability the image is AI-generated
    threshold: float
    model_name: str
    model_version: str
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def inspect_image(data: bytes) -> ImageInfo:
    """Validate by CONTENT (magic bytes + full decode), never by file name or browser-supplied MIME type."""
    if not data:
        raise InvalidImageError("The file is empty.")
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            if fmt not in ALLOWED_FORMATS:
                raise InvalidImageError(f"Unsupported image format {fmt!r}. Use JPEG, PNG, WEBP, GIF or BMP.")
            im.verify()  # structural check
        with Image.open(io.BytesIO(data)) as im:
            im.load()  # full pixel decode: catches truncated/corrupt files that pass verify()
            return ImageInfo(fmt, im.width, im.height, im.mode, len(data))
    except InvalidImageError:
        raise
    except Image.DecompressionBombError as e:
        raise InvalidImageError("The image is too large to process safely.") from e
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as e:
        raise InvalidImageError("The file looks like an image but cannot be decoded.") from e


def _tf_compatible_bytes(data: bytes, info: ImageInfo) -> bytes:
    if info.format in TF_NATIVE_FORMATS:
        return data
    with Image.open(io.BytesIO(data)) as im:
        buf = io.BytesIO()
        im.save(buf, format="PNG")  # lossless: same pixels the browser shows
        return buf.getvalue()


class ChampionModel:
    """Holds the loaded Keras model and its serving config. Create once per process; thread-safe predict()."""

    source = "local"  # where inference runs (recorded in the prediction trail)

    def __init__(self, model_path: str | Path, config_path: str | Path | None = None,
                 threshold: float | None = None):
        import keras  # imported here so importing this module (e.g. for validation helpers) stays cheap

        model_path = Path(model_path)
        config_path = Path(config_path) if config_path else model_path.with_name("serving_config.json")
        self.config = json.loads(config_path.read_text())
        t0 = time.perf_counter()
        self.model = keras.models.load_model(model_path)
        self.load_seconds = time.perf_counter() - t0
        self.img_size = int(self.config["img_size"])
        self.resize_mode = self.config.get("resize_mode", "stretch")
        self.threshold = float(threshold if threshold is not None else self.config["threshold"])
        self.model_name = str(self.config.get("registered_model") or self.config.get("model_name", model_path.stem))
        self.model_version = str(self.config.get("version", "local"))
        self._lock = threading.Lock()
        self.predict_bytes(self._warmup_bytes())  # build the graph now, not on the first user request

    def _warmup_bytes(self) -> bytes:
        buf = io.BytesIO()
        Image.new("RGB", (self.img_size, self.img_size)).save(buf, format="PNG")
        return buf.getvalue()

    def predict_bytes(self, data: bytes, info: ImageInfo | None = None) -> Prediction:
        import tensorflow as tf

        from visionguard.data import decode_and_resize

        info = info or inspect_image(data)
        t0 = time.perf_counter()
        x = decode_and_resize(tf.constant(_tf_compatible_bytes(data, info)), self.img_size, self.resize_mode)
        x = tf.cast(x, tf.float32)[tf.newaxis]
        with self._lock:  # Keras models are not guaranteed thread-safe for concurrent calls
            p_ai = float(self.model(x, training=False).numpy().reshape(-1)[0])
        label = "ai" if p_ai >= self.threshold else "real"
        return Prediction(label=label, confidence=p_ai if label == "ai" else 1.0 - p_ai, p_ai=p_ai,
                          threshold=self.threshold, model_name=self.model_name, model_version=self.model_version,
                          latency_ms=(time.perf_counter() - t0) * 1000)

    def describe(self) -> dict:
        return {"model_name": self.model_name, "model_version": self.model_version, "threshold": self.threshold,
                "img_size": self.img_size, "resize_mode": self.resize_mode,
                "source_run": self.config.get("model_name"), "load_seconds": round(self.load_seconds, 2)}


def main() -> None:
    import argparse
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    ap = argparse.ArgumentParser(description="Classify images with the exported champion model.")
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("MODEL_PATH", "models/champion.keras")))
    a = ap.parse_args()

    t0 = time.perf_counter()
    model = ChampionModel(a.model)
    print(f"loaded {model.model_name} v{model.model_version} once in {time.perf_counter() - t0:.2f}s "
          f"(threshold {model.threshold:.4f}, {model.img_size}px, {model.resize_mode})")
    for path in a.images:
        try:
            p = model.predict_bytes(path.read_bytes())
            print(f"{path}: label={p.label} confidence={p.confidence:.3f} p_ai={p.p_ai:.4f} "
                  f"latency={p.latency_ms:.0f}ms")
        except InvalidImageError as e:
            print(f"{path}: REJECTED — {e}")


if __name__ == "__main__":
    main()
