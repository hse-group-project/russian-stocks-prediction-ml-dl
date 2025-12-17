from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)
from sqlalchemy.orm import Session
from sqlalchemy import delete, select
from src.api.database import Candles, Indices, RequestLog
from src.api.session import get_db

router = APIRouter(prefix="/api")


@router.post("/forward")
async def forward(db: Session = Depends(get_db)):
    pass


@router.get("/candles")
def get_candles(
    left_date: str, right_date: str, ticker: str, db: Session = Depends(get_db)
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


@router.get("/indices")
def get_indices(
    left_date: str, right_date: str, index_code: str, db: Session = Depends(get_db)
):
    try:
        res = select(Indices).where(
            Indices.date >= left_date,
            Indices.date <= right_date,
            Indices.index_code == index_code,
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
        await db.commit()
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
