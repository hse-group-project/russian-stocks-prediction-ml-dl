from fastapi import FastAPI
from src.api.routes import router
from src.api.database import Base
from src.api.session import engine

Base.metadata.create_all(bind=engine)

app = FastAPI(title="ML Service")
app.include_router(router)
