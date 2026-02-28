from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.config import settings
from app.api.routes import fencers, upload, bouts

app = FastAPI(title="Fencing Analyzer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fencers.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(bouts.router, prefix="/api")

# Serve uploaded videos directly in dev
uploads_dir = "/app/uploads"
os.makedirs(uploads_dir, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=uploads_dir), name="uploads")


@app.get("/health")
def health():
    return {"status": "ok"}
