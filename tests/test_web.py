from __future__ import annotations

import io
from collections.abc import Callable

from flask import Flask
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

from visionguard.web.db import Database, PredictionRecord


def _records(app: Flask) -> list[PredictionRecord]:
    database: Database = app.extensions["vg_db"]
    records, total = database.page(page=1, per_page=100)
    assert total == len(records)
    return records


def _upload(client: FlaskClient, data: bytes, filename: str) -> TestResponse:
    return client.post(
        "/predict",
        data={"image": (io.BytesIO(data), filename)},
        content_type="multipart/form-data",
    )


def test_index_prediction_result_and_health(
    app: Flask, make_image_bytes: Callable[[str, tuple[int, int], str], bytes]
) -> None:
    client = app.test_client()

    assert client.get("/").status_code == 200
    response = _upload(client, make_image_bytes("RGB", (14, 8), "PNG"), "../../etc/passwd.png")

    assert response.status_code == 200
    assert "Likely AI-generated" in response.get_data(as_text=True)
    records = _records(app)
    assert len(records) == 1
    record = records[0]
    assert record.label == "ai"
    assert record.confidence == 0.91
    assert record.model_name == "fake-champion"
    assert record.model_version == "test-1"
    assert record.filename == "etc_passwd.png"
    assert "/" not in record.filename
    assert "\\" not in record.filename

    assert client.get(f"/result/{record.id}").status_code == 200
    assert client.get("/result/9999").status_code == 404

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json is not None
    assert health.json["database"] == "ok"
    assert health.json["debug"] is False


def test_failed_uploads_do_not_create_prediction_records(
    app: Flask, make_image_bytes: Callable[[str, tuple[int, int], str], bytes]
) -> None:
    client = app.test_client()

    tiff = _upload(client, make_image_bytes("RGB", (8, 8), "TIFF"), "unsupported.tiff")
    undecodable = _upload(client, b"not an image", "invalid.png")
    missing_file = client.post("/predict", data={}, content_type="multipart/form-data")

    assert tiff.status_code == 415
    assert undecodable.status_code == 400
    assert missing_file.status_code == 400
    assert _records(app) == []


def test_oversize_upload_does_not_create_prediction_record(
    app_factory: Callable[[float], Flask], make_image_bytes: Callable[[str, tuple[int, int], str], bytes]
) -> None:
    app = app_factory(0.01)

    response = _upload(app.test_client(), make_image_bytes("RGB", (128, 128), "BMP"), "large.bmp")

    assert response.status_code == 413
    assert _records(app) == []


def test_history_lists_predictions_newest_first(
    app: Flask, make_image_bytes: Callable[[str, tuple[int, int], str], bytes]
) -> None:
    client = app.test_client()
    first = _upload(client, make_image_bytes("RGB", (12, 8), "PNG"), "first.png")
    second = _upload(client, make_image_bytes("RGB", (12, 8), "PNG"), "second.png")

    assert first.status_code == 200
    assert second.status_code == 200
    history = client.get("/history")
    text = history.get_data(as_text=True)

    assert history.status_code == 200
    assert text.index("second.png") < text.index("first.png")
