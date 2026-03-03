import os
import uuid
import subprocess
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session as DBSession

from app.database import get_db
from app.models.bout import Bout, Session
from app.models.fencer import Fencer
from app.schemas.bout import BoutUploadResponse
from app.config import settings

router = APIRouter(prefix="/upload", tags=["upload"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 2048  # 2 GB


@router.post("/", response_model=BoutUploadResponse, status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    fencer_id: int = Form(...),
    db: DBSession = Depends(get_db),
):
    # Validate fencer exists
    fencer = db.get(Fencer, fencer_id)
    if not fencer:
        raise HTTPException(status_code=404, detail="Fencer not found")

    # Validate file type
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Create or get a session for today
    from datetime import date, datetime, timezone
    today = date.today()
    db_session = (
        db.query(Session)
        .filter(Session.fencer_id == fencer_id)
        .filter(Session.date >= datetime(today.year, today.month, today.day, tzinfo=timezone.utc))
        .first()
    )
    if not db_session:
        db_session = Session(fencer_id=fencer_id)
        db.add(db_session)
        db.flush()

    # Save file locally (S3 upload wired in later)
    upload_dir = "/app/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    video_key = f"{uuid.uuid4()}{ext}"
    dest_path = os.path.join(upload_dir, video_key)

    with open(dest_path, "wb") as f:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            f.write(chunk)

    # Extract thumbnail from first frame
    video_key_no_ext = os.path.splitext(video_key)[0]
    thumb_path = f"/app/uploads/thumb_{video_key_no_ext}.jpg"
    subprocess.run([
        "ffmpeg", "-v", "error", "-i", dest_path,
        "-vf", "select=eq(n\\,0)", "-vframes", "1",
        "-q:v", "3", thumb_path, "-y"
    ], check=True)

    # Create bout record in "configuring" state — pipeline not dispatched yet
    bout = Bout(
        session_id=db_session.id,
        video_key=video_key,
        video_url=f"/uploads/{video_key}",
        status="configuring",
    )
    db.add(bout)
    db.commit()
    db.refresh(bout)

    return BoutUploadResponse(bout_id=bout.id, task_id=None, status="configuring")
