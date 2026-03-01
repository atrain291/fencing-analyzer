import os

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.bout import Bout
from app.schemas.bout import BoutRead

router = APIRouter(prefix="/bouts", tags=["bouts"])


@router.get("/{bout_id}", response_model=BoutRead)
def get_bout(bout_id: int, db: Session = Depends(get_db)):
    bout = db.get(Bout, bout_id)
    if not bout:
        raise HTTPException(status_code=404, detail="Bout not found")
    return bout


@router.delete("/{bout_id}", status_code=204)
def delete_bout(bout_id: int, db: Session = Depends(get_db)):
    bout = db.get(Bout, bout_id)
    if not bout:
        raise HTTPException(status_code=404, detail="Bout not found")

    if bout.task_id and bout.status in ("queued", "processing"):
        from app.tasks import celery_app
        celery_app.control.revoke(bout.task_id, terminate=True)

    video_path = f"/app/uploads/{bout.video_key}"
    if os.path.exists(video_path):
        os.remove(video_path)

    db.delete(bout)
    db.commit()
    return Response(status_code=204)


@router.get("/{bout_id}/status")
def get_bout_status(bout_id: int, db: Session = Depends(get_db)):
    bout = db.get(Bout, bout_id)
    if not bout:
        raise HTTPException(status_code=404, detail="Bout not found")
    return {
        "bout_id": bout_id,
        "status": bout.status,
        "pipeline_progress": bout.pipeline_progress,
        "error": bout.error,
    }
