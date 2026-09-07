"""
Application entrypoint. Run locally with:
    uvicorn app.main:app --reload

This wires together the routers described in Section 3.2.2, loads the three
Logistic Regression models at start-up (Section 3.2.7), and configures CORS
so the Next.js frontend (a separate origin) can call the API (Section 3.2.1).
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import init_db
from app.routers import admin, appointments, auth, prediction, telemedicine
from app.services.prediction_service import load_models


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # for local/dev; use Alembic migrations in production
    load_models()
    yield


app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(prediction.router)
app.include_router(appointments.router)
app.include_router(appointments.doctors_router)
app.include_router(admin.router)
app.include_router(telemedicine.router)


@app.get("/api/v1/health", tags=["Health"])
def health_check():
    return {"status": "ok"}
