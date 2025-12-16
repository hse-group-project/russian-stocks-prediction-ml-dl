from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()


class RequestLog(Base):
    __tablename__ = "request_logs"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.now(datetime.timezone.utc))
    input_type = Column(String)  # "json" | "image"
    input_size = Column(Integer)
    processing_time = Column(Float)
    status = Column(String)  # "success" | "model_failed" | "bad_request"
    result_preview = Column(Text, nullable=True)
