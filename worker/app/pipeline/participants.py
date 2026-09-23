"""Conservative per-bout association; missing tracks never exchange roles."""
import math


class ParticipantMap:
    """Seed exactly two tracks left-to-right, then keep their IDs for this bout.

    This is a positional convention, not verification of the uploaded fencer's
    identity. A crowded shot needs explicit participant selection in a later UI.
    Track fragmentation intentionally produces missing data instead of guessing.
    """

    def __init__(self):
        self.track_ids = None

    def select(self, track_ids, boxes) -> tuple[int | None, int | None]:
        if len(track_ids) != len(boxes):
            raise ValueError("Track IDs and boxes must have the same length")
        ids = [int(value) for value in track_ids]
        if len(set(ids)) != len(ids):
            return None, None
        if self.track_ids is None:
            if len(ids) != 2:
                return None, None
            if any(len(box) != 4 or not all(math.isfinite(float(v)) for v in box) for box in boxes):
                return None, None
            ordered = sorted(range(2), key=lambda i: float(boxes[i][0]) + float(boxes[i][2]))
            self.track_ids = tuple(ids[i] for i in ordered)
        by_id = {track_id: index for index, track_id in enumerate(ids)}
        return tuple(by_id.get(track_id) for track_id in self.track_ids)
