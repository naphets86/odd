from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os

from app.core.config import settings
from app.core.database import engine, Base
from app.core.redis_client import init_redis, close_redis
from app.routers import auth, contacts, messages, calls, ws


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_redis()
    yield
    await close_redis()


app = FastAPI(
    title="Chat Me API",
    version="1.0.0",
    description="Backend für Chat Me – Signatur-Chiffre E2E-Verschlüsselung",
    lifespan=lifespan,
)

# CORS
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router)
app.include_router(contacts.router)
app.include_router(messages.router)
app.include_router(calls.router)
app.include_router(ws.router)


# Frontend – index.html liegt in ../frontend/ (ein Verzeichnis über dem Backend)
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
FRONTEND_DIR = os.path.abspath(FRONTEND_DIR)

if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_spa():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Chat Me API"}


@app.get("/api/turn-credentials")
async def turn_credentials():
    """Gibt TURN-Credentials für WebRTC zurück."""
    return {
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"},
            {
                "urls":       f"turn:{settings.TURN_REALM}:3478",
                "username":   settings.TURN_USER,
                "credential": settings.TURN_PASSWORD,
            },
        ]
    }
