from fastapi import FastAPI
from src.api.database import RequestLog
from src.api.routes import router
from src.api.session import engine

RequestLog.__table__.create(bind=engine, checkfirst=True)

app = FastAPI(title="ML Service")
app.include_router(router)
