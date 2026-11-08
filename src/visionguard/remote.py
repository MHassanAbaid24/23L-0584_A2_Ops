"""Remote inference client for the deployed VisionGuard model API."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from visionguard.inference import ImageInfo, Prediction, _tf_compatible_bytes, inspect_image


class RemoteModelError(RuntimeError):
    """The hosted model cannot provide a prediction."""

    def __init__(self, message: str, status_code: int = 503):
        super().__init__(message)
        self.status_code = status_code


def _json_object(payload: bytes) -> dict[str, object]:
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RemoteModelError("model service unavailable") from error
    if not isinstance(decoded, dict):
        raise RemoteModelError("model service unavailable")
    return decoded


def _required_string(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise RemoteModelError("model API health response is invalid")
    return value


def _required_number(payload: Mapping[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RemoteModelError("model API response is invalid")
    return float(value)


def _multipart_image(image: bytes) -> tuple[bytes, str]:
    boundary = f"----VisionGuard{uuid.uuid4().hex}"
    body = b"".join((
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="image"; filename="image.png"\r\n',
        b"Content-Type: image/png\r\n\r\n",
        image,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ))
    return body, boundary


class RemotePredictor:
    """Preprocess uploads locally, then request inference from the hosted model API."""

    source = "vercel"

    def __init__(self, api_url: str, api_key: str, timeout: float):
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        if not self.api_url or not self.api_key or self.timeout <= 0:
            raise ValueError("remote inference requires an API URL, API key, and positive timeout")

        started = time.perf_counter()
        health = self._fetch_health(self.timeout)
        self.health_seconds = time.perf_counter() - started
        self.model_name = _required_string(health, "model_name")
        self.model_version = _required_string(health, "model_version")
        self.source_run = _required_string(health, "source_run")
        self.threshold = _required_number(health, "threshold")
        img_size = health.get("img_size")
        if isinstance(img_size, bool) or not isinstance(img_size, int) or img_size <= 0:
            raise RemoteModelError("model API health response is invalid")
        self.img_size = img_size
        resize_mode = health.get("resize_mode", "stretch")
        if not isinstance(resize_mode, str) or resize_mode not in {"stretch", "pad"}:
            raise RemoteModelError("model API health response is invalid")
        self.resize_mode = resize_mode
        self.runtime = _required_string(health, "runtime")

    def _fetch_health(self, timeout: float) -> dict[str, object]:
        request = Request(f"{self.api_url}/api/health", method="GET")
        try:
            with urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RemoteModelError("model service unavailable")
                return _json_object(response.read())
        except RemoteModelError:
            raise
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise RemoteModelError("model service unavailable") from error

    def health_check(self, timeout: float) -> bool:
        try:
            self._fetch_health(timeout)
        except RemoteModelError:
            return False
        return True

    def predict_bytes(self, data: bytes, info: ImageInfo | None = None) -> Prediction:
        import tensorflow as tf

        from visionguard.data import decode_and_resize

        info = info or inspect_image(data)
        image = decode_and_resize(
            tf.constant(_tf_compatible_bytes(data, info)), self.img_size, self.resize_mode
        )
        png = tf.io.encode_png(image).numpy()
        body, boundary = _multipart_image(png)
        request = Request(
            f"{self.api_url}/api/predict",
            data=body,
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "X-API-Key": self.api_key,
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout) as response:
                if response.status != 200:
                    raise RemoteModelError("model service unavailable")
                payload = _json_object(response.read())
        except HTTPError as error:
            if error.code == 401:
                raise RemoteModelError("model API rejected the key", status_code=502) from error
            if 400 <= error.code < 500:
                try:
                    message = _required_string(_json_object(error.read()), "error")
                except RemoteModelError:
                    message = "model API request failed"
                raise RemoteModelError(message) from error
            raise RemoteModelError("model service unavailable") from error
        except RemoteModelError:
            raise
        except (URLError, TimeoutError, OSError) as error:
            raise RemoteModelError("model service unavailable") from error

        label = _required_string(payload, "label")
        return Prediction(
            label=label,
            confidence=_required_number(payload, "confidence"),
            p_ai=_required_number(payload, "p_ai"),
            threshold=_required_number(payload, "threshold"),
            model_name=_required_string(payload, "model_name"),
            model_version=_required_string(payload, "model_version"),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def describe(self) -> dict[str, str | float | int]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "threshold": self.threshold,
            "img_size": self.img_size,
            "resize_mode": self.resize_mode,
            "source_run": self.source_run,
            "load_seconds": round(self.health_seconds, 2),
            "api_url": self.api_url,
            "runtime": self.runtime,
        }
