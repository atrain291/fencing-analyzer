"""Decode upright video frames without replacing presentation time with FPS."""
from dataclasses import dataclass
from typing import Any, Iterator


@dataclass
class DecodedFrame:
    index: int
    timestamp_ms: float
    image: Any


def upright_bgr(image, rotation: int):
    """PyAV display-matrix angles are counterclockwise."""
    import numpy as np

    if rotation % 90:
        raise ValueError(f"Unsupported video rotation {rotation}; normalize the video first")
    return np.ascontiguousarray(np.rot90(image, rotation // 90)) if rotation else image


def iter_video_frames(path: str) -> Iterator[DecodedFrame]:
    """Use the container timeline (seconds), retaining VFR and initial offsets.

    Software decoding is intentional: the previous raw NVDEC pipe discarded
    timestamps. Callers closing the generator early also close the container.
    """
    import av

    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream in {path}")
        stream = container.streams.video[0]
        origin = container.start_time / av.time_base if container.start_time is not None else None
        if origin is None and stream.start_time is not None:
            origin = float(stream.start_time * stream.time_base)
        previous = None
        count = 0
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError(f"Frame {index} has no presentation timestamp")
            seconds = float(frame.pts * frame.time_base)
            if origin is None:
                origin = seconds
            timestamp_ms = round((seconds - origin) * 1000, 6)
            if timestamp_ms < 0 or (previous is not None and timestamp_ms <= previous):
                raise ValueError(f"Non-monotonic presentation timestamp at frame {index}")
            previous = timestamp_ms
            image = upright_bgr(frame.to_ndarray(format="bgr24"), frame.rotation)
            count += 1
            yield DecodedFrame(index, timestamp_ms, image)
        if not count:
            raise ValueError(f"No decoded frames in {path}")
