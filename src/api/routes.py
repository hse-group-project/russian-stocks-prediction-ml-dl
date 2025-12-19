from datetime import datetime, timezone
import time
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import delete, select
from src.core.model import resulting, make_prediction
from src.api.database import Candles, Companies, RequestLog
from src.api.session import get_db

router = APIRouter(prefix="/api")


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
def forward(request: ForwardRequest, db: Session = Depends(get_db)):
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
        log_entry.result_preview = str(res)[
            :500
        ]  # или json.dumps(res)[:500] если нужно

        db.add(log_entry)
        db.commit()
        return res

    except Exception as e:
        processing_time = time.perf_counter() - start_time
        log_entry.processing_time = processing_time
        log_entry.status = "model_failed"
        log_entry.result_preview = None

        db.add(log_entry)
        db.commit()

        raise HTTPException(status_code=500, detail=str(e))


@router.post("/fake_predict")
def fake_predict(request: FakePredictRequest):
    try:
        res = make_prediction(request.ticker, request.interval)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/companies")
def get_companies(db: Session = Depends(get_db)):
    try:
        res = select(Companies.ticker)
        result = db.execute(res)
        companies = result.scalars().all()
        return companies
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/candles")
def get_candles(
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
async def delete_history(db: Session = Depends(get_db)):
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

        stats = {}

        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
