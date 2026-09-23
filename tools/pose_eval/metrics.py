"""Separate detection coverage from manually validated joint/identity accuracy."""
import math
from statistics import mean


def evaluate(records, annotations, model_id, confidence_threshold=0.3):
    summary = {
        "frames_with_two_or_more_people_fraction": (
            sum(len(record["detections"]) >= 2 for record in records) / len(records) if records else None),
        "mean_visible_joint_error_px": None,
        "visible_joint_recall": None,
        "annotated_visible_joints": 0,
        "matched_visible_joints": 0,
        "labeled_id_switches": None,
        "association_note": "Accuracy requires explicit per-model person/track labels; coverage includes background people.",
    }
    if annotations is None:
        return summary
    by_frame = {record["frame_index"]: record for record in records}
    previous_ids, errors, total, associated_observations, switches = {}, [], 0, 0, 0
    for annotation in sorted(annotations, key=lambda a: a["frame_index"]):
        index = annotation["frame_index"]
        if index not in by_frame:
            raise ValueError(f"Annotation frame {index} is outside this evaluated clip")
        record = by_frame[index]
        detections = {d["track_id"]: d for d in record["detections"] if d.get("track_id") is not None}
        for person in annotation["people"]:
            if model_id not in person["track_ids"]:
                continue  # Not annotated for this model; different from a confirmed miss (null).
            track_id = person["track_ids"][model_id]
            if track_id is not None and track_id not in detections:
                raise ValueError(f"Annotated track {track_id} absent for {model_id} at frame {index}; use null for a miss")
            if track_id is not None:
                associated_observations += 1
                previous = previous_ids.get(person["person_id"])
                switches += int(previous is not None and previous != track_id)
                previous_ids[person["person_id"]] = track_id
            detection = detections.get(track_id, {})
            for name, expected in person["keypoints"].items():
                if not expected["visible"]:
                    continue
                total += 1
                point = detection.get("keypoints", {}).get(name)
                if point is None or point["confidence"] < confidence_threshold:
                    continue
                dx = (point["x"] - expected["x"]) * record["width"]
                dy = (point["y"] - expected["y"]) * record["height"]
                errors.append(math.hypot(dx, dy))
    summary.update(
        mean_visible_joint_error_px=mean(errors) if errors else None,
        visible_joint_recall=len(errors) / total if total else None,
        annotated_visible_joints=total,
        matched_visible_joints=len(errors),
        labeled_id_switches=switches if associated_observations else None,
    )
    return summary
