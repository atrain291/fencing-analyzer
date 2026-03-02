import os

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models.bout import Bout
from app.models.analysis import Frame, BladeState
from app.schemas.bout import BoutRead

router = APIRouter(prefix="/bouts", tags=["bouts"])


class BboxModel(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class ConfigureROIRequest(BaseModel):
    fencer_bbox: BboxModel | None = None
    opponent_bbox: BboxModel | None = None


@router.get("/{bout_id}", response_model=BoutRead)
def get_bout(bout_id: int, db: Session = Depends(get_db)):
    bout = (
        db.query(Bout)
        .options(joinedload(Bout.frames).joinedload(Frame.blade_state))
        .filter(Bout.id == bout_id)
        .first()
    )
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


@router.get("/{bout_id}/thumbnail")
def get_thumbnail(bout_id: int, db: Session = Depends(get_db)):
    bout = db.get(Bout, bout_id)
    if not bout:
        raise HTTPException(status_code=404, detail="Bout not found")
    video_key_no_ext = os.path.splitext(bout.video_key or "")[0]
    thumb_path = f"/app/uploads/thumb_{video_key_no_ext}.jpg"
    if not os.path.exists(thumb_path):
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(thumb_path, media_type="image/jpeg")


@router.post("/{bout_id}/roi")
def configure_roi(bout_id: int, body: ConfigureROIRequest, db: Session = Depends(get_db)):
    bout = db.get(Bout, bout_id)
    if not bout:
        raise HTTPException(status_code=404, detail="Bout not found")
    if bout.status != "configuring":
        raise HTTPException(status_code=400, detail="Bout is not awaiting configuration")

    bout.fencer_bbox = body.fencer_bbox.model_dump() if body.fencer_bbox else None
    bout.opponent_bbox = body.opponent_bbox.model_dump() if body.opponent_bbox else None
    bout.status = "queued"
    db.commit()

    from app.tasks import dispatch_pipeline
    task = dispatch_pipeline(bout_id, f"/app/uploads/{bout.video_key}")
    bout.task_id = task.id
    db.commit()

    return {"bout_id": bout_id, "status": "queued", "task_id": task.id}
