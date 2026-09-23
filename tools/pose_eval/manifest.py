"""Resolve and validate local evaluation inputs before loading any model."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re


@dataclass(frozen=True)
class Clip:
    id: str
    video: Path
    start_s: float
    end_s: float
    annotations: Path | None = None


def _time(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return float(value)


def load_manifest(path: Path | str) -> list[Clip]:
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("clips"), list) or not data["clips"]:
        raise ValueError("Manifest must contain a non-empty clips array")
    clips, seen = [], set()
    for item in data["clips"]:
        if not isinstance(item, dict):
            raise ValueError("Each clip must be an object")
        clip_id = item.get("id", "")
        if not isinstance(clip_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", clip_id):
            raise ValueError("Clip id must contain only letters, digits, underscores, or hyphens")
        if clip_id in seen:
            raise ValueError(f"Duplicate clip id: {clip_id}")
        seen.add(clip_id)
        start, end = _time(item.get("start_s", 0), "start_s"), _time(item.get("end_s"), "end_s")
        if end <= start:
            raise ValueError(f"Clip {clip_id}: end_s must be greater than start_s")
        if not isinstance(item.get("video"), str) or not item["video"]:
            raise ValueError(f"Clip {clip_id} needs a video path")
        video = (path.parent / item["video"]).resolve()
        if not video.is_file():
            raise ValueError(f"Video not found: {video}")
        annotations = item.get("annotations")
        if annotations is not None:
            if not isinstance(annotations, str) or not annotations:
                raise ValueError("annotations must be a path or null")
            annotations = (path.parent / annotations).resolve()
            if not annotations.is_file():
                raise ValueError(f"Annotations not found: {annotations}")
            load_annotations(annotations)
        clips.append(Clip(clip_id, video, start, end, annotations))
    return clips


def load_annotations(path: Path | None):
    """Identity association is explicit per model, using IDs visible in the report."""
    if path is None:
        return None
    from . import MODEL_IDS
    from worker.app.pipeline.pose import KEYPOINT_NAMES

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Annotations must be a list of labeled frames")
    seen_frames = set()
    for frame in data:
        if not isinstance(frame, dict):
            raise ValueError("Each annotation frame must be an object")
        index = frame.get("frame_index")
        if type(index) is not int or index < 0 or index in seen_frames:
            raise ValueError("Annotation frame_index must be unique and non-negative")
        seen_frames.add(index)
        people = frame.get("people")
        if not isinstance(people, list):
            raise ValueError("Annotation frame requires people array")
        seen_people, seen_tracks = set(), set()
        for person in people:
            if not isinstance(person, dict) or not isinstance(person.get("person_id"), str) or not person["person_id"]:
                raise ValueError("Each annotated person needs a person_id")
            if person["person_id"] in seen_people:
                raise ValueError("Duplicate annotated person in one frame")
            seen_people.add(person["person_id"])
            ids = person.get("track_ids")
            if not isinstance(ids, dict) or set(ids) - set(MODEL_IDS):
                raise ValueError("track_ids must map supported model IDs to a track ID or null")
            for model_id, track_id in ids.items():
                if track_id is not None:
                    if type(track_id) is not int or track_id < 0 or (model_id, track_id) in seen_tracks:
                        raise ValueError("Each model track ID must be unique per annotated frame")
                    seen_tracks.add((model_id, track_id))
            joints = person.get("keypoints")
            if not isinstance(joints, dict) or set(joints) - set(KEYPOINT_NAMES):
                raise ValueError("Annotated keypoints must use COCO joint names")
            for point in joints.values():
                if not isinstance(point, dict) or type(point.get("visible")) is not bool:
                    raise ValueError("Each annotated joint requires visible: true/false")
                if point["visible"]:
                    for axis in ("x", "y"):
                        coordinate = _time(point.get(axis), axis)
                        if coordinate > 1:
                            raise ValueError("Annotation coordinates must be normalized to [0, 1]")
    return data
