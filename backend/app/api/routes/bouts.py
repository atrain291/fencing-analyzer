import os
import statistics

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models.bout import Bout, Session as SessionModel
from app.models.analysis import Action, Analysis, Frame, BladeState
from app.schemas.bout import BoutRead, BoutSummary

router = APIRouter(prefix="/bouts", tags=["bouts"])


class BboxModel(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class DetectionSelection(BaseModel):
    frame_index: int
    detection_index: int


class ConfigureROIRequest(BaseModel):
    fencer_bbox: BboxModel | None = None
    opponent_bbox: BboxModel | None = None
    fencer_detection: DetectionSelection | None = None
    opponent_detection: DetectionSelection | None = None


@router.get("/", response_model=list[BoutSummary])
def list_bouts(fencer_id: int, db: Session = Depends(get_db)):
    bouts = (
        db.query(Bout)
        .join(SessionModel, Bout.session_id == SessionModel.id)
        .filter(SessionModel.fencer_id == fencer_id)
        .order_by(Bout.created_at.desc())
        .all()
    )
    return bouts


@router.get("/{bout_id}", response_model=BoutRead)
def get_bout(bout_id: int, db: Session = Depends(get_db)):
    bout = (
        db.query(Bout)
        .options(joinedload(Bout.analysis))
        .filter(Bout.id == bout_id)
        .first()
    )
    if not bout:
        raise HTTPException(status_code=404, detail="Bout not found")
    return bout


@router.get("/{bout_id}/frames")
def get_bout_frames(bout_id: int, db: Session = Depends(get_db)):
    """Return all frames + blade states + actions for overlay rendering."""
    bout = db.get(Bout, bout_id)
    if not bout:
        raise HTTPException(status_code=404, detail="Bout not found")
    frames = (
        db.query(Frame)
        .options(joinedload(Frame.blade_state))
        .filter(Frame.bout_id == bout_id)
        .order_by(Frame.timestamp_ms)
        .all()
    )
    actions = (
        db.query(Action)
        .filter(Action.bout_id == bout_id)
        .order_by(Action.start_ms)
        .all()
    )
    from app.schemas.bout import FrameRead, ActionRead
    return {
        "frames": [FrameRead.model_validate(f).model_dump() for f in frames],
        "actions": [ActionRead.model_validate(a).model_dump() for a in actions],
    }


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


@router.get("/{bout_id}/preview")
def get_preview(bout_id: int, db: Session = Depends(get_db)):
    bout = db.get(Bout, bout_id)
    if not bout:
        raise HTTPException(status_code=404, detail="Bout not found")
    if bout.status == "failed":
        return {"status": "failed", "error": bout.error}
    if bout.preview_data is None:
        return {"status": "processing"}
    return {"status": "ready", "preview_data": bout.preview_data}


@router.post("/{bout_id}/preview")
def trigger_preview(bout_id: int, db: Session = Depends(get_db)):
    bout = db.get(Bout, bout_id)
    if not bout:
        raise HTTPException(status_code=404, detail="Bout not found")
    if bout.status not in ("configuring", "failed", "preview_ready"):
        raise HTTPException(status_code=400, detail="Bout is not in a state that allows preview")
    # Guard against double-dispatch
    if bout.preview_data is not None and bout.status == "preview_ready":
        return {"bout_id": bout_id, "status": "preview_ready", "preview_data": bout.preview_data}
    bout.status = "previewing"
    bout.preview_data = None
    db.commit()
    from app.tasks import dispatch_preview
    task = dispatch_preview(bout_id, f"/app/uploads/{bout.video_key}")
    bout.task_id = task.id
    db.commit()
    return {"bout_id": bout_id, "status": "previewing", "task_id": task.id}


@router.post("/{bout_id}/roi")
def configure_roi(bout_id: int, body: ConfigureROIRequest, db: Session = Depends(get_db)):
    bout = db.get(Bout, bout_id)
    if not bout:
        raise HTTPException(status_code=404, detail="Bout not found")
    if bout.status not in ("configuring", "preview_ready", "failed"):
        raise HTTPException(status_code=400, detail="Bout is not awaiting configuration")

    # Clean slate for re-analysis of failed bouts
    if bout.status == "failed":
        db.query(BladeState).filter(
            BladeState.frame_id.in_(
                db.query(Frame.id).filter(Frame.bout_id == bout_id)
            )
        ).delete(synchronize_session=False)
        db.query(Action).filter(Action.bout_id == bout_id).delete(synchronize_session=False)
        db.query(Frame).filter(Frame.bout_id == bout_id).delete(synchronize_session=False)
        bout.error = None
        bout.pipeline_progress = None

    # Resolve detection selections to bboxes from preview data
    if body.fencer_detection and bout.preview_data:
        frame_data = bout.preview_data["frames"][body.fencer_detection.frame_index]
        det = frame_data["detections"][body.fencer_detection.detection_index]
        bout.fencer_bbox = det["bbox"]
    elif body.fencer_bbox:
        bout.fencer_bbox = body.fencer_bbox.model_dump()
    else:
        bout.fencer_bbox = None

    if body.opponent_detection and bout.preview_data:
        frame_data = bout.preview_data["frames"][body.opponent_detection.frame_index]
        det = frame_data["detections"][body.opponent_detection.detection_index]
        bout.opponent_bbox = det["bbox"]
    elif body.opponent_bbox:
        bout.opponent_bbox = body.opponent_bbox.model_dump()
    else:
        bout.opponent_bbox = None
    bout.status = "queued"
    db.commit()

    from app.tasks import dispatch_pipeline
    task = dispatch_pipeline(bout_id, f"/app/uploads/{bout.video_key}")
    bout.task_id = task.id
    db.commit()

    return {"bout_id": bout_id, "status": "queued", "task_id": task.id}


def _coeff_of_variation_score(values: list[float]) -> float:
    """Return 100 * (1 - CV) clamped to [0, 100].  CV = std_dev / mean."""
    if len(values) < 2:
        return 100.0
    mean = statistics.mean(values)
    if mean == 0:
        return 100.0
    cv = statistics.stdev(values) / mean
    return max(0.0, min(100.0, 100.0 * (1.0 - cv)))


@router.get("/{bout_id}/drill-report")
def get_drill_report(bout_id: int, db: Session = Depends(get_db)):
    bout = db.get(Bout, bout_id)
    if not bout:
        raise HTTPException(status_code=404, detail="Bout not found")
    if bout.status not in ("done", "complete"):
        raise HTTPException(status_code=400, detail="Pipeline has not completed for this bout")

    actions = (
        db.query(Action)
        .filter(Action.bout_id == bout_id)
        .order_by(Action.start_ms)
        .all()
    )

    if not actions:
        raise HTTPException(status_code=404, detail="No actions detected for this bout")

    # ---- Action breakdown ----
    from collections import defaultdict
    by_type: dict[str, list[float]] = defaultdict(list)
    for a in actions:
        duration = float(a.end_ms - a.start_ms)
        by_type[a.type].append(duration)

    action_breakdown = {}
    all_consistency_scores = []
    for action_type, durations in by_type.items():
        avg_dur = statistics.mean(durations)
        score = _coeff_of_variation_score(durations)
        action_breakdown[action_type] = {
            "count": len(durations),
            "avg_duration_ms": round(avg_dur, 1),
            "consistency_score": round(score, 1),
        }
        all_consistency_scores.append(score)

    total_actions = len(actions)
    total_duration_ms = actions[-1].end_ms - actions[0].start_ms
    tempo = (total_actions / (total_duration_ms / 1000.0)) if total_duration_ms > 0 else 0.0

    # ---- Rhythm score: consistency of gaps between consecutive actions ----
    gaps = []
    for i in range(1, len(actions)):
        gap = float(actions[i].start_ms - actions[i - 1].end_ms)
        gaps.append(gap)
    rhythm_score = _coeff_of_variation_score(gaps) if gaps else 100.0

    # ---- Tempo score: CV of rolling 5-action window action rates ----
    window_size = 5
    window_rates = []
    for i in range(len(actions) - window_size + 1):
        window_actions = actions[i : i + window_size]
        window_dur_ms = window_actions[-1].end_ms - window_actions[0].start_ms
        if window_dur_ms > 0:
            rate = window_size / (window_dur_ms / 1000.0)
            window_rates.append(rate)
    tempo_score = _coeff_of_variation_score(window_rates) if window_rates else 100.0

    # ---- Overall score ----
    avg_consistency = statistics.mean(all_consistency_scores) if all_consistency_scores else 100.0
    overall_score = 0.4 * rhythm_score + 0.3 * tempo_score + 0.3 * avg_consistency

    # ---- Auto-detect drill type from action distribution ----
    footwork_types = {"advance", "retreat", "lunge"}
    footwork_count = sum(len(by_type[t]) for t in footwork_types if t in by_type)
    drill_type = "footwork" if footwork_count >= total_actions * 0.5 else "mixed"

    return {
        "drill_type": drill_type,
        "total_actions": total_actions,
        "total_duration_ms": total_duration_ms,
        "tempo": round(tempo, 2),
        "action_breakdown": action_breakdown,
        "rhythm_score": round(rhythm_score, 1),
        "tempo_score": round(tempo_score, 1),
        "overall_score": round(overall_score, 1),
    }
