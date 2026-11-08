from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import keras
import numpy as np
import pytest
import tensorflow as tf

from visionguard.data import decode_and_resize
from visionguard.inference import ChampionModel, InvalidImageError, inspect_image


@pytest.mark.parametrize(("mode", "image_format"), [("L", "PNG"), ("RGBA", "PNG"), ("P", "PNG")])
def test_decode_and_resize_converts_image_modes_to_rgb(
    make_image_bytes: Callable[[str, tuple[int, int], str], bytes], mode: str, image_format: str
) -> None:
    decoded = decode_and_resize(tf.constant(make_image_bytes(mode, (13, 7), image_format)), img_size=19).numpy()

    assert decoded.shape == (19, 19, 3)
    assert decoded.dtype == np.uint8
    assert decoded.min() >= 0
    assert decoded.max() <= 255


def test_decode_and_resize_pad_preserves_aspect_ratio(
    make_image_bytes: Callable[[str, tuple[int, int], str], bytes]
) -> None:
    data = make_image_bytes("RGB", (20, 10), "PNG")

    padded = decode_and_resize(tf.constant(data), img_size=20, resize_mode="pad").numpy()
    stretched = decode_and_resize(tf.constant(data), img_size=20, resize_mode="stretch").numpy()

    assert np.all(padded[0] == 0)
    assert np.all(padded[-1] == 0)
    assert np.any(padded[10] != 0)
    assert np.any(stretched[0] != 0)
    assert np.any(stretched[-1] != 0)


def test_decode_and_resize_uses_content_not_a_misleading_extension(
    make_image_bytes: Callable[[str, tuple[int, int], str], bytes]
) -> None:
    png_bytes_with_jpg_name = make_image_bytes("RGB", (9, 5), "PNG")

    decoded = decode_and_resize(tf.constant(png_bytes_with_jpg_name), img_size=11).numpy()

    assert decoded.shape == (11, 11, 3)
    assert decoded.dtype == np.uint8


def test_inspect_image_rejects_unsupported_tiff(make_image_bytes: Callable[[str, tuple[int, int], str], bytes]) -> None:
    with pytest.raises(InvalidImageError, match="Unsupported image format"):
        inspect_image(make_image_bytes("RGB", (8, 8), "TIFF"))


def test_inspect_image_rejects_text_bytes() -> None:
    with pytest.raises(InvalidImageError, match="cannot be decoded"):
        inspect_image(b"this is not an image")


def test_inspect_image_rejects_truncated_jpeg(
    make_image_bytes: Callable[[str, tuple[int, int], str], bytes]
) -> None:
    jpeg = make_image_bytes("RGB", (32, 16), "JPEG")

    with pytest.raises(InvalidImageError, match="cannot be decoded"):
        inspect_image(jpeg[: len(jpeg) // 2])


def test_inspect_image_accepts_webp(make_image_bytes: Callable[[str, tuple[int, int], str], bytes]) -> None:
    info = inspect_image(make_image_bytes("RGB", (12, 6), "WEBP"))

    assert info.format == "WEBP"
    assert (info.width, info.height) == (12, 6)


def test_champion_model_predicts_from_tiny_saved_keras_model(
    tmp_path: Path, make_image_bytes: Callable[[str, tuple[int, int], str], bytes]
) -> None:
    model_path = tmp_path / "tiny.keras"
    config_path = tmp_path / "serving_config.json"
    model = keras.Sequential(
        [
            keras.Input(shape=(32, 32, 3)),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(1, activation="sigmoid", kernel_initializer="zeros", bias_initializer="zeros"),
        ]
    )
    model.save(model_path)
    config_path.write_text(
        json.dumps(
            {
                "img_size": 32,
                "resize_mode": "stretch",
                "threshold": 0.6,
                "registered_model": "tiny-champion",
                "version": "test",
            }
        )
    )

    champion = ChampionModel(model_path, config_path)
    prediction = champion.predict_bytes(make_image_bytes("RGB", (12, 6), "PNG"))

    assert prediction.label in {"ai", "real"}
    assert 0.5 <= prediction.confidence <= 1
    assert prediction.label == ("ai" if prediction.p_ai >= prediction.threshold else "real")
    assert prediction.confidence == pytest.approx(
        prediction.p_ai if prediction.label == "ai" else 1 - prediction.p_ai
    )
