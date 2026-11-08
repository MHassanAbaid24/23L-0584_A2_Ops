"""VisionGuard's TFLite-backed Vercel API."""

from __future__ import annotations

import hmac
import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

MAX_UPLOAD_BYTES = 4 * 1024 * 1024
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024
SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "GIF", "BMP"})
MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def _load_config() -> dict[str, object]:
    with (MODEL_DIR / "serving_config.json").open(encoding="utf-8") as config_file:
        return json.load(config_file)


CONFIG = _load_config()
INTERPRETER = Interpreter(model_path=str(MODEL_DIR / "champion.tflite"))
INTERPRETER.allocate_tensors()
INPUT_INDEX = int(INTERPRETER.get_input_details()[0]["index"])
OUTPUT_INDEX = int(INTERPRETER.get_output_details()[0]["index"])
INTERPRETER_LOCK = threading.Lock()
API_KEY = os.environ.get("VG_API_KEY", "")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES


def _error(message: str, status_code: int):
    return jsonify(error=message), status_code


@app.errorhandler(RequestEntityTooLarge)
def _request_too_large(_: RequestEntityTooLarge):
    return _error("Image exceeds the 4 MB limit.", 413)


@app.errorhandler(405)
def _method_not_allowed(_: object):
    return _error("Method not allowed.", 405)


def _verify_api_key(value: str | None) -> bool:
    return bool(API_KEY) and value is not None and hmac.compare_digest(value, API_KEY)


def _preprocess(data: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = image.format
            if image_format not in SUPPORTED_FORMATS:
                raise ValueError("Unsupported image format.")
            image.seek(0)
            rgb = image.convert("RGB")
            img_size = int(CONFIG["img_size"])
            if rgb.size != (img_size, img_size):
                rgb = rgb.resize((img_size, img_size), Image.Resampling.BILINEAR)
            return np.asarray(rgb, dtype=np.float32)[np.newaxis, ...]
    except (UnidentifiedImageError, OSError, ValueError) as error:
        if str(error) == "Unsupported image format.":
            raise ValueError("Unsupported image format.") from error
        raise ValueError("Image could not be decoded.") from error


@app.get("/api/health")
def health():
    return jsonify(
        status="ok",
        model_name=CONFIG.get("registered_model", CONFIG["model_name"]),
        model_version=CONFIG["version"],
        source_run=CONFIG["model_name"],
        threshold=float(CONFIG["threshold"]),
        img_size=int(CONFIG["img_size"]),
        runtime="tflite",
    )


@app.post("/api/predict")
def predict():
    if not _verify_api_key(request.headers.get("X-API-Key")):
        return _error("Unauthorized.", 401)

    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return _error("Missing image file.", 400)
    data = upload.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return _error("Image exceeds the 4 MB limit.", 413)
    if not data:
        return _error("Image could not be decoded.", 400)
    try:
        image_input = _preprocess(data)
    except ValueError as error:
        return _error(str(error), 415 if str(error) == "Unsupported image format." else 400)

    started_at = time.perf_counter()
    with INTERPRETER_LOCK:
        INTERPRETER.set_tensor(INPUT_INDEX, image_input)
        INTERPRETER.invoke()
        p_ai = float(INTERPRETER.get_tensor(OUTPUT_INDEX).reshape(-1)[0])
    latency_ms = (time.perf_counter() - started_at) * 1000
    threshold = float(CONFIG["threshold"])
    label = "ai" if p_ai >= threshold else "real"
    return jsonify(
        label=label,
        confidence=p_ai if label == "ai" else 1.0 - p_ai,
        p_ai=p_ai,
        threshold=threshold,
        model_name=CONFIG.get("registered_model", CONFIG["model_name"]),
        model_version=CONFIG["version"],
        source_run=CONFIG["model_name"],
        latency_ms=latency_ms,
        runtime="tflite",
    )
