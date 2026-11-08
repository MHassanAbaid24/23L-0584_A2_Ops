from __future__ import annotations

import io
from collections.abc import Callable
from itertools import count
from pathlib import Path

import pytest
from flask import Flask
from PIL import Image

from visionguard.inference import ImageInfo, Prediction
from visionguard.web import create_app
from visionguard.web.config import Settings


class FakePredictor:
    source = "fake"

    def predict_bytes(self, data: bytes, info: ImageInfo | None = None) -> Prediction:
        return Prediction(
            label="ai",
            confidence=0.91,
            p_ai=0.91,
            threshold=0.6,
            model_name="fake-champion",
            model_version="test-1",
            latency_ms=2.5,
        )

    def describe(self) -> dict[str, str | float | int]:
        return {
            "model_name": "fake-champion",
            "model_version": "test-1",
            "threshold": 0.6,
            "img_size": 32,
            "resize_mode": "stretch",
            "source_run": "fake-run",
            "load_seconds": 0,
        }


def _image_bytes(mode: str = "RGB", size: tuple[int, int] = (16, 8), image_format: str = "PNG") -> bytes:
    colors: dict[str, int | tuple[int, int, int] | tuple[int, int, int, int]] = {
        "L": 127,
        "RGB": (220, 120, 40),
        "RGBA": (220, 120, 40, 80),
        "P": 1,
    }
    image = Image.new(mode, size, color=colors[mode])
    if mode == "P":
        image.putpalette([0, 0, 0, 220, 120, 40] + [0] * 762)
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.fixture
def make_image_bytes() -> Callable[[str, tuple[int, int], str], bytes]:
    return _image_bytes


@pytest.fixture
def fake_predictor() -> FakePredictor:
    return FakePredictor()


@pytest.fixture
def app_factory(tmp_path: Path, fake_predictor: FakePredictor) -> Callable[[float], Flask]:
    database_number = count()

    def build(max_upload_mb: float = 10) -> Flask:
        settings = Settings(
            model_path="unused.keras",
            model_config_path=None,
            threshold=None,
            database_url=f"sqlite:///{tmp_path / f'predictions-{next(database_number)}.sqlite'}",
            max_upload_mb=max_upload_mb,
            secret_key="test-secret",
            history_page_size=20,
            log_level="WARNING",
        )
        return create_app(settings, fake_predictor)

    return build


@pytest.fixture
def app(app_factory: Callable[[float], Flask]) -> Flask:
    return app_factory()
