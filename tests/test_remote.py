from __future__ import annotations

import io
import json
import socket
from collections.abc import Iterator
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import cast

import pytest
from flask import Flask
from flask.testing import FlaskClient
from PIL import Image
from werkzeug.test import TestResponse

from visionguard.remote import RemoteModelError
from visionguard.web import create_app
from visionguard.web.config import Settings
from visionguard.web.db import Database, PredictionRecord


class ModelAPIServer(ThreadingHTTPServer):
    expected_key: str
    received_image_size: tuple[int, int] | None
    received_image_format: str | None


class ModelAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/api/health":
            self._send_json(404, {"error": "not found"})
            return
        self._send_json(
            200,
            {
                "status": "ok",
                "model_name": "remote-champion",
                "model_version": "remote-7",
                "source_run": "remote-run",
                "threshold": 0.73,
                "img_size": 24,
                "resize_mode": "stretch",
                "runtime": "tflite",
            },
        )

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/predict":
            self._send_json(404, {"error": "not found"})
            return
        server = cast(ModelAPIServer, self.server)
        if self.headers.get("X-API-Key") != server.expected_key:
            self._send_json(401, {"error": "invalid API key"})
            return
        content_type = self.headers.get("Content-Type", "")
        boundary_marker = "boundary="
        if boundary_marker not in content_type:
            self._send_json(400, {"error": "missing multipart image"})
            return
        boundary = content_type.split(boundary_marker, maxsplit=1)[1].encode()
        length = int(self.headers["Content-Length"])
        request_body = self.rfile.read(length)
        image_bytes: bytes | None = None
        for part in request_body.split(b"--" + boundary):
            if b'name="image"' not in part or b"\r\n\r\n" not in part:
                continue
            image_bytes = part.split(b"\r\n\r\n", maxsplit=1)[1].rsplit(b"\r\n", maxsplit=1)[0]
            break
        if image_bytes is None:
            self._send_json(400, {"error": "missing multipart image"})
            return
        with Image.open(io.BytesIO(image_bytes)) as image:
            server.received_image_size = image.size
            server.received_image_format = image.format
        self._send_json(
            200,
            {
                "label": "ai",
                "confidence": 0.88,
                "p_ai": 0.88,
                "threshold": 0.73,
                "model_name": "remote-champion",
                "model_version": "remote-7",
                "source_run": "remote-run",
                "latency_ms": 1.5,
                "runtime": "tflite",
            },
        )


@dataclass
class StandInAPI:
    server: ModelAPIServer
    thread: Thread
    stopped: bool = False

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        if self.stopped:
            return
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.stopped = True


@pytest.fixture
def stand_in_api() -> Iterator[StandInAPI]:
    server = ModelAPIServer(("127.0.0.1", 0), ModelAPIHandler)
    server.expected_key = "test-key"
    server.received_image_size = None
    server.received_image_format = None
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api = StandInAPI(server, thread)
    try:
        yield api
    finally:
        api.stop()


def _settings(tmp_path: Path, api_url: str, api_key: str) -> Settings:
    return Settings(
        model_path="unused.keras",
        model_config_path=None,
        threshold=None,
        database_url=f"sqlite:///{tmp_path / 'remote.sqlite'}",
        max_upload_mb=10,
        secret_key="test-secret",
        history_page_size=20,
        log_level="WARNING",
        inference_mode="remote",
        model_api_url=api_url,
        model_api_key=api_key,
        model_api_timeout=1,
    )


def _upload(client: FlaskClient, image: bytes) -> TestResponse:
    return client.post(
        "/predict",
        data={"image": (io.BytesIO(image), "upload.png")},
        content_type="multipart/form-data",
    )


def _records(app: Flask) -> list[PredictionRecord]:
    database: Database = app.extensions["vg_db"]
    records, total = database.page(page=1, per_page=100)
    assert total == len(records)
    return records


def test_remote_prediction_stores_vercel_source_and_preprocesses_upload(
    tmp_path: Path, stand_in_api: StandInAPI, make_image_bytes
) -> None:
    app = create_app(_settings(tmp_path, stand_in_api.url, "test-key"))

    response = _upload(app.test_client(), make_image_bytes("RGB", (13, 7), "PNG"))

    assert response.status_code == 200
    record = _records(app)[0]
    assert record.source == "vercel"
    assert record.model_name == "remote-champion"
    assert record.model_version == "remote-7"
    assert (record.threshold, record.p_ai, record.confidence) == (0.73, 0.88, 0.88)
    assert stand_in_api.server.received_image_size == (24, 24)
    assert stand_in_api.server.received_image_format == "PNG"
    health = app.test_client().get("/health")
    assert health.json is not None
    assert health.json["inference_mode"] == "remote"
    assert health.json["api_url"] == stand_in_api.url
    assert health.json["api_health"] is True


def test_remote_wrong_key_returns_bad_gateway_without_record(
    tmp_path: Path, stand_in_api: StandInAPI, make_image_bytes
) -> None:
    app = create_app(_settings(tmp_path, stand_in_api.url, "wrong-key"))

    response = _upload(app.test_client(), make_image_bytes("RGB", (13, 7), "PNG"))

    assert response.status_code == 502
    assert "model API rejected the key" in response.get_data(as_text=True)
    assert _records(app) == []


def test_remote_api_down_returns_service_unavailable_without_record(
    tmp_path: Path, stand_in_api: StandInAPI, make_image_bytes
) -> None:
    app = create_app(_settings(tmp_path, stand_in_api.url, "test-key"))
    stand_in_api.stop()

    response = _upload(app.test_client(), make_image_bytes("RGB", (13, 7), "PNG"))

    assert response.status_code == 503
    assert "model service unavailable" in response.get_data(as_text=True)
    assert _records(app) == []


def test_remote_mode_fails_fast_when_health_is_unreachable(tmp_path: Path) -> None:
    with socket.socket() as unavailable_socket:
        unavailable_socket.bind(("127.0.0.1", 0))
        host, port = unavailable_socket.getsockname()
    with pytest.raises(RemoteModelError, match="model service unavailable"):
        create_app(_settings(tmp_path, f"http://{host}:{port}", "test-key"))
