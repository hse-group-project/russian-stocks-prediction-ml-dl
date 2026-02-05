from datetime import datetime, timezone
import os
import time
from dotenv import load_dotenv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Header, Request, status
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import delete, select
from src.core.model import resulting, make_prediction
from src.api.database import Candles, Companies, RequestLog
from src.api.session import get_db

router = APIRouter(prefix="/api")

load_dotenv()

### Пока что реализованы не все модели из бд (пока что написано то, что нужно в задаче)


class ForwardRequest(BaseModel):
    ticker: str
    left_date: str
    right_date: str
    train_period: int
    val_period: int
    test_period: int
    step: int
    top_n_features: int
    metric_optuna: str


class FakePredictRequest(BaseModel):
    ticker: str
    interval: int


@router.post("/forward")
async def forward(request: ForwardRequest, req: Request, db: Session = Depends(get_db)):
    content_type = req.headers.get("content-type", "").lower()
    if "application/json" not in content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="bad request"
        )

    start_time = time.perf_counter()

    input_json_str = request.model_dump_json()  # Pydantic v2
    # Или: input_json_str = request.json()  # если Pydantic v1

    log_entry = RequestLog(
        input_type="json",
        input_size=len(input_json_str.encode("utf-8")),
        status="pending",
        timestamp=datetime.now(timezone.utc),
    )

    try:
        res = resulting(
            ticker=request.ticker,
            left_date=request.left_date,
            right_date=request.right_date,
            train_period=request.train_period,
            val_period=request.val_period,
            test_period=request.test_period,
            step=request.step,
            top_n_features=request.top_n_features,
            metric_optuna=request.metric_optuna,
        )

        processing_time = time.perf_counter() - start_time
        log_entry.processing_time = processing_time
        log_entry.status = "success"
        log_entry.result_preview = str(res)[:500]

        db.add(log_entry)
        db.commit()
        return res

    except Exception:
        processing_time = time.perf_counter() - start_time
        log_entry.processing_time = processing_time
        log_entry.status = "model_failed"
        log_entry.result_preview = None

        db.add(log_entry)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The model was unable to process the data",
        )


@router.post("/test_predict")
async def test_predict(request: FakePredictRequest):
    try:
        res = make_prediction(request.ticker, request.interval)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/companies")
async def get_companies(db: Session = Depends(get_db)):
    try:
        res = select(Companies.ticker)
        result = db.execute(res)
        companies = result.scalars().all()
        return companies
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/candles")
async def get_candles(
    ticker: str, left_date: str, right_date: str, db: Session = Depends(get_db)
):
    try:
        res = select(Candles).where(
            Candles.datetime >= left_date,
            Candles.datetime <= right_date,
            Candles.ticker == ticker,
        )
        result = db.execute(res)
        candles = result.scalars().all()
        return candles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_history(db: Session = Depends(get_db)):
    try:
        result = db.execute(select(RequestLog))
        logs = result.scalars().all()

        if not logs:
            return {"message": "Not logs yet"}

        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/history")
async def delete_history(
    admin_token: str = Header(..., alias="X-Admin-Token"), db: Session = Depends(get_db)
):
    expected_token = os.getenv("ADMIN_DELETE_TOKEN")
    if not expected_token:
        raise HTTPException(
            status_code=500,
            detail="Server misconfiguration: ADMIN_DELETE_TOKEN not set",
        )

    if admin_token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token")

    try:
        result = db.execute(delete(RequestLog))
        db.commit()
        return {"message": f"Deleted {result.rowcount} records"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    try:
        result = db.execute(select(RequestLog).where(RequestLog.status == "success"))
        logs = result.scalars().all()

        if not logs:
            return {"message": "Not successful logs yet"}

        processing_times = [
            log.processing_time for log in logs if log.processing_time is not None
        ]
        input_sizes = [log.input_size for log in logs if log.input_size is not None]

        if not processing_times:
            return {"message": "No processing time data"}

        pt = np.array(processing_times)
        size = np.array(input_sizes) if input_sizes else np.array([])

        stats = {
            "processing_time_sec": {
                "mean": float(np.mean(pt)),
                "median_50%": float(np.percentile(pt, 50)),
                "p95": float(np.percentile(pt, 95)),
                "p99": float(np.percentile(pt, 99)),
                "count": len(pt),
            },
            "input_size_bytes": {
                "mean": float(np.mean(size)) if size.size else 0,
                "median_50%": float(np.percentile(size, 50)) if size.size else 0,
                "p95": float(np.percentile(size, 95)) if size.size else 0,
                "p99": float(np.percentile(size, 99)) if size.size else 0,
                "count": len(size),
            },
        }

        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
