"""VisionGuard web app (Task 4) with the prediction trail (Task 5).

Production:  gunicorn -c gunicorn.conf.py "visionguard.web:create_app()"
Routes:      GET /  (upload) · POST /predict (result) · GET /result/<id> · GET /history · GET /health
"""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Protocol

from flask import Flask, abort, jsonify, render_template, request
from PIL import Image
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

from visionguard.inference import ImageInfo, InvalidImageError, Prediction, inspect_image
from visionguard.web.config import Settings
from visionguard.web.db import Database, PredictionRecord

log = logging.getLogger("visionguard.web")


class Predictor(Protocol):
    source: str

    def predict_bytes(self, data: bytes, info: ImageInfo | None = None) -> Prediction: ...
    def describe(self) -> dict: ...


class InMemoryUploadRequest(Flask.request_class):
    """Keep uploaded files in RAM. Werkzeug normally spools uploads >500 KB to a temp file on disk; the brief
    forbids keeping the image on disk, and MAX_CONTENT_LENGTH already bounds the memory used."""

    def _get_file_stream(self, total_content_length, content_type, filename=None, content_length=None):
        return io.BytesIO()



def _thumbnail_data_uri(data: bytes, size: int = 320) -> str:
    """A small preview for the result page, sent inline in the HTML and never written anywhere."""
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _clean_filename(name: str | None) -> str:
    cleaned = secure_filename(name or "")[:120]
    return cleaned or "upload"


def create_app(settings: Settings | None = None, predictor: Predictor | None = None) -> Flask:
    settings = settings or Settings.from_env()
    logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = Flask(__name__)
    app.request_class = InMemoryUploadRequest
    app.config.update(SECRET_KEY=settings.secret_key, MAX_CONTENT_LENGTH=int(settings.max_upload_mb * 1024 * 1024))

    if settings.database_url.startswith("sqlite:///"):
        Path(settings.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    db = Database(settings.database_url)
    db.create_schema()

    if predictor is None:
        from visionguard.inference import ChampionModel

        model = ChampionModel(settings.model_path, settings.model_config_path, settings.threshold)
        predictor = model
        log.info("model loaded once at start-up: %s", model.describe())

    app.extensions["vg_db"] = db
    app.extensions["vg_predictor"] = predictor
    app.extensions["vg_settings"] = settings

    @app.context_processor
    def _inject():
        return {"model_info": predictor.describe(), "max_upload_mb": settings.max_upload_mb}

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.post("/predict")
    def predict():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            abort(400, description="Choose an image file to upload.")
        filename = _clean_filename(upload.filename)
        data = upload.read()
        try:
            info = inspect_image(data)
        except InvalidImageError as e:
            log.info("rejected upload %r: %s", filename, e)
            abort(415 if "Unsupported" in str(e) else 400, description=str(e))
        t0 = time.perf_counter()
        pred = predictor.predict_bytes(data, info)
        record = db.add(PredictionRecord(
            filename=filename, image_format=info.format, width=info.width, height=info.height,
            file_size=info.size_bytes, label=pred.label, confidence=pred.confidence, p_ai=pred.p_ai,
            threshold=pred.threshold, model_name=pred.model_name, model_version=pred.model_version,
            latency_ms=pred.latency_ms, source=getattr(predictor, "source", "local")))
        log.info("prediction id=%s file=%s label=%s conf=%.3f total=%.0fms", record.id, filename, pred.label,
                 pred.confidence, (time.perf_counter() - t0) * 1000)
        return render_template("result.html", r=record, preview=_thumbnail_data_uri(data))

    @app.get("/result/<int:record_id>")
    def result(record_id: int):
        record = db.get(record_id)
        if record is None:
            abort(404, description=f"No prediction with id {record_id}.")
        return render_template("result.html", r=record, preview=None)

    @app.get("/history")
    def history():
        page = max(1, request.args.get("page", 1, type=int))
        rows, total = db.page(page, settings.history_page_size)
        pages = max(1, -(-total // settings.history_page_size))
        return render_template("history.html", rows=rows, page=page, pages=pages, total=total, stats=db.stats())

    @app.get("/health")
    def health():
        db_ok = db.ping()
        body = {"status": "ok" if db_ok else "degraded", "model": predictor.describe(),
                "database": "ok" if db_ok else "unreachable", "debug": app.debug}
        return jsonify(body), 200 if db_ok else 503

    @app.errorhandler(HTTPException)
    def http_error(e: HTTPException):
        if isinstance(e, RequestEntityTooLarge):
            e.description = f"The file is larger than the {settings.max_upload_mb:g} MB limit."
        if request.path == "/health":
            return jsonify(error=e.name, detail=e.description), e.code
        return render_template("error.html", code=e.code, title=e.name, message=e.description), e.code

    @app.errorhandler(Exception)
    def unhandled(e: Exception):
        log.exception("unhandled error on %s", request.path)
        return render_template("error.html", code=500, title="Something went wrong",
                               message="The server hit an unexpected error. Please try again."), 500

    return app
