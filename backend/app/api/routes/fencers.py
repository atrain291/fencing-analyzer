from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.fencer import Fencer
from app.schemas.fencer import FencerCreate, FencerRead

router = APIRouter(prefix="/fencers", tags=["fencers"])


@router.post("/", response_model=FencerRead, status_code=201)
def create_fencer(body: FencerCreate, db: Session = Depends(get_db)):
    fencer = Fencer(name=body.name)
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
