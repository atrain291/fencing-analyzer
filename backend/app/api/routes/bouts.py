from fastapi import APIRouter, Depends, HTTPException
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
