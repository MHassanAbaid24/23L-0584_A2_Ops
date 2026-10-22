"""Task 5 — prediction trail. One row per classified upload: what was decided, when, by which model, how confidently.

DATABASE_URL selects the backend: PostgreSQL in Docker Compose (postgresql+psycopg://…), SQLite for local runs.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True,
                                                 default=lambda: datetime.now(UTC))
    filename: Mapped[str] = mapped_column(String(120))          # sanitized; the image itself is never stored
    image_format: Mapped[str] = mapped_column(String(10))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    file_size: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(10), index=True)  # "ai" | "real"
    confidence: Mapped[float] = mapped_column(Float)
    p_ai: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(120))
    model_version: Mapped[str] = mapped_column(String(40))
    latency_ms: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(20), default="local")  # where inference ran: local | remote-api


class Database:
    def __init__(self, url: str):
        kwargs = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
        self.engine: Engine = create_engine(url, **kwargs)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001 — health check reports any failure as "down"
            return False

    def add(self, record: PredictionRecord) -> PredictionRecord:
        with Session(self.engine, expire_on_commit=False) as s, s.begin():
            s.add(record)
        return record

    def get(self, record_id: int) -> PredictionRecord | None:
        with Session(self.engine) as s:
            return s.get(PredictionRecord, record_id)

    def page(self, page: int, per_page: int) -> tuple[list[PredictionRecord], int]:
        with Session(self.engine) as s:
            total = s.scalar(select(func.count()).select_from(PredictionRecord)) or 0
            rows = s.scalars(select(PredictionRecord).order_by(PredictionRecord.id.desc())
                             .offset((page - 1) * per_page).limit(per_page)).all()
            return list(rows), total

    def stats(self) -> dict:
        with Session(self.engine) as s:
            rows = s.execute(select(PredictionRecord.label, func.count()).group_by(PredictionRecord.label)).all()
            return {label: n for label, n in rows}
