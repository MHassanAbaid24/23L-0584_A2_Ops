"""Task 4.4 — every setting comes from environment variables (12-factor); nothing environment-specific in code."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass


def _float_or_none(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


@dataclass(frozen=True)
class Settings:
    model_path: str
    model_config_path: str | None
    threshold: float | None          # optional override; default comes from serving_config.json
    database_url: str
    max_upload_mb: float
    secret_key: str
    history_page_size: int
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_path=os.environ.get("MODEL_PATH", "models/champion.keras"),
            model_config_path=os.environ.get("MODEL_CONFIG_PATH") or None,
            threshold=_float_or_none(os.environ.get("THRESHOLD")),
            database_url=os.environ.get("DATABASE_URL", "sqlite:///instance/visionguard.db"),
            max_upload_mb=float(os.environ.get("MAX_UPLOAD_MB", "10")),
            # Only signs flash messages; a random per-process key is acceptable when unset.
            secret_key=os.environ.get("SECRET_KEY") or secrets.token_hex(32),
            history_page_size=int(os.environ.get("HISTORY_PAGE_SIZE", "20")),
            log_level=os.environ.get("LOG_LEVEL", "INFO"),
        )
