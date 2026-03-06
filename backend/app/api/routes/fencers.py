import os

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.fencer import Fencer
from app.models.bout import Session as SessionModel, Bout
from app.schemas.fencer import FencerCreate, FencerRead

router = APIRouter(prefix="/fencers", tags=["fencers"])


@router.post("/", response_model=FencerRead, status_code=201)
def create_fencer(body: FencerCreate, db: Session = Depends(get_db)):
    existing = db.query(Fencer).filter(Fencer.name == body.name.strip()).first()
    if existing:
        raise HTTPException(status_code=409, detail="A fencer with this name already exists")
    fencer = Fencer(name=body.name.strip())
    db.add(fencer)
    db.commit()
    db.refresh(fencer)
    return fencer


@router.get("/", response_model=list[FencerRead])
def list_fencers(db: Session = Depends(get_db)):
    return db.query(Fencer).order_by(Fencer.created_at.desc()).all()


@router.get("/{fencer_id}", response_model=FencerRead)
def get_fencer(fencer_id: int, db: Session = Depends(get_db)):
    fencer = db.get(Fencer, fencer_id)
    if not fencer:
        raise HTTPException(status_code=404, detail="Fencer not found")
    return fencer


@router.delete("/{fencer_id}", status_code=204)
def delete_fencer(fencer_id: int, db: Session = Depends(get_db)):
    fencer = db.get(Fencer, fencer_id)
    if not fencer:
        raise HTTPException(status_code=404, detail="Fencer not found")

    # Clean up video files, thumbnails, and preview images for all bouts
    for session in fencer.sessions:
        for bout in session.bouts:
            if bout.task_id and bout.status in ("queued", "processing"):
                try:
                    from app.tasks import celery_app
                    celery_app.control.revoke(bout.task_id, terminate=True)
                except Exception:
                    pass

            if bout.video_key:
                video_path = f"/app/uploads/{bout.video_key}"
                if os.path.exists(video_path):
                    os.remove(video_path)

                video_key_no_ext = os.path.splitext(bout.video_key)[0]
                thumb_path = f"/app/uploads/thumb_{video_key_no_ext}.jpg"
                if os.path.exists(thumb_path):
                    os.remove(thumb_path)

            # Clean up preview images
            if bout.preview_data and "frames" in bout.preview_data:
                for frame_data in bout.preview_data["frames"]:
                    image_key = frame_data.get("image_key", "")
                    if image_key:
                        preview_path = f"/app/uploads/{image_key}"
                        if os.path.exists(preview_path):
                            os.remove(preview_path)

    # Cascade: Fencer -> Sessions -> Bouts -> Actions/Frames/Analysis/BladeStates
    db.delete(fencer)
    db.commit()
    return Response(status_code=204)
