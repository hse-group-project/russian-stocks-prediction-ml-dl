from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    UploadFile,
    File,
    Header,
    status,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
from src.api.database import RequestLog
from src.api.session import get_db, get_current_admin
from src.api.auth import verify_delete_token  # для PRO-требования токена
from src.api.model import your_ml_model  # замените на вашу модель
from typing import Optional, Union, Dict, Any
import json
import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np

router = APIRouter()


@router.get("/")
async def root():
    return {"message": "Hello, world!"}


@router.post("/forward")
async def forward():
    pass


@router.get("/history")
async def get_history(db: AsyncSession = Depends(get_db)):
    try:
        result = await db.execute(select(RequestLog))
        logs = result.scalars().all()
        return logs
    except Exception as e:
        return e


@router.delete("/history")
async def delete_history():
    pass


@router.get("/stats")
async def get_stats(db: AsyncSession = Depends(get_db)):
    try:
        pass
    except Exception as e:
        return e