from datetime import datetime
from pydantic import BaseModel


class FencerCreate(BaseModel):
    name: str


class FencerRead(BaseModel):
    model_config = {"from_attributes": True}

    id: int
    name: str
    created_at: datetime
