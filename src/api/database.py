from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Text,
    TIMESTAMP,
    Numeric,
    Boolean,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import BIGINT
from datetime import datetime, timezone

Base = declarative_base()


class RequestLog(Base):
    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    input_type = Column(String)  # "json" | "image"
    input_size = Column(Integer)
    processing_time = Column(Float)
    status = Column(String)  # "success" | "model_failed" | "bad_request"
    result_preview = Column(Text, nullable=True)


class Companies(Base):
    __tablename__ = "companies"

    id = Column(BIGINT, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False)
    name = Column(Text, nullable=False)
    figi = Column(String(50), nullable=False)
    currency = Column(String(10), nullable=False)
    lot = Column(Numeric(15, 6), nullable=False)
    min_price_increment = Column(Numeric(10, 6), nullable=False)
    sector = Column(Text, nullable=False)
    created_at = Column(
        TIMESTAMP, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        TIMESTAMP, nullable=False, default=lambda: datetime.now(timezone.utc)
    )


class Candles(Base):
    __tablename__ = "candles"

    id = Column(BIGINT, primary_key=True, autoincrement=True)
    ticker = Column(String(20), nullable=False)
    datetime = Column(TIMESTAMP, nullable=False)
    open = Column(Numeric(15, 6), nullable=False)
    high = Column(Numeric(15, 6), nullable=False)
    low = Column(Numeric(15, 6), nullable=False)
    close = Column(Numeric(15, 6), nullable=False)
    volume = Column(BIGINT, nullable=False)
    is_complete = Column(Boolean, nullable=False, default=False)
    created_at = Column(
        TIMESTAMP, nullable=False, default=lambda: datetime.now(timezone.utc)
    )
