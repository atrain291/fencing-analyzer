"""Run one model/clip and preserve the evidence needed to reproduce the result."""
from contextlib import closing
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
from statistics import mean, median
import subprocess
from time import perf_counter

from worker.app.pipeline.video import iter_video_frames
from .manifest import load_annotations
from .metrics import evaluate


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    versions = {}
    for name in ("av", "numpy", "torch", "torchvision", "ultralytics", "rfdetr", "supervision"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "packages": versions, "git_commit": commit, "git_dirty": dirty}


def run_clip(clip, adapter, output, *, warmup=3):
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be a non-negative integer")
    output = Path(output) / clip.id
    destination = output / f'{adapter.metadata["model_id"]}.json'
    if destination.exists():
        raise FileExistsError(f"Refusing to replace {destination}; choose another output directory")
    annotations = load_annotations(clip.annotations)
    source_hash = sha256_file(clip.video)
    adapter.reset()
    records, latencies, tracking_times = [], [], []
    synchronize = getattr(adapter, "synchronize", lambda: None)
    tracking = getattr(adapter, "track", lambda detections, width, height: detections)
    warmup_seconds = 0.0
    began = perf_counter()
    with closing(iter_video_frames(str(clip.video))) as frames:
        for frame in frames:
            if frame.timestamp_ms < clip.start_s * 1000:
                continue
            if frame.timestamp_ms >= clip.end_s * 1000:
                break
            if not records:
                warmup_start = perf_counter()
                for _ in range(warmup):
                    adapter.predict(frame.image)
                synchronize()
                adapter.reset()
                getattr(adapter, "reset_memory_stats", lambda: None)()
                warmup_seconds = perf_counter() - warmup_start
            synchronize()
            start = perf_counter()
            detections = adapter.predict(frame.image)
            synchronize()
            prediction_ms = (perf_counter() - start) * 1000
            start = perf_counter()
            height, width = frame.image.shape[:2]
            detections = tracking(detections, width, height)
            tracking_ms = (perf_counter() - start) * 1000
            latencies.append(prediction_ms)
            tracking_times.append(tracking_ms)
            records.append({"frame_index": frame.index, "timestamp_ms": frame.timestamp_ms,
                            "width": width, "height": height, "detections": detections,
                            "prediction_ms": prediction_ms, "tracking_ms": tracking_ms})
    processing_seconds = perf_counter() - began - warmup_seconds
    if not records:
        raise ValueError(f"Clip {clip.id} contains no frames in [{clip.start_s}, {clip.end_s})")
    p95_index = min(len(latencies) - 1, max(0, math.ceil(len(latencies) * 0.95) - 1))
    result = {
        "schema_version": 1, "status": "complete", "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {"name": clip.video.name, "path": str(clip.video), "sha256": source_hash, "size_bytes": clip.video.stat().st_size,
                   "clip_id": clip.id, "start_s": clip.start_s, "end_s": clip.end_s},
        "provenance": {**environment(), **adapter.metadata},
        "summary": {"frame_count": len(records), "mean_prediction_ms": mean(latencies),
                    "median_prediction_ms": median(latencies), "p95_prediction_ms": sorted(latencies)[p95_index],
                    "mean_tracking_ms": mean(tracking_times), "warmup_frames": warmup,
                    "warmup_seconds": warmup_seconds, "processing_seconds": processing_seconds,
                    "peak_cuda_allocated_bytes": getattr(adapter, "peak_memory", lambda: None)(),
                    "timing_scope": "Prediction includes preprocessing, model inference, output conversion; tracking is separate. "
                                    "Processing includes sequential decoding from the file start, prediction and tracking; "
                                    "excludes model setup, warmup, hashing, metrics and exports."},
        "metrics": evaluate(records, annotations, adapter.metadata["model_id"]),
        "annotation_sha256": sha256_file(clip.annotations) if clip.annotations else None,
        "frames": records,
    }
    serialized = json.dumps(result, allow_nan=False)
    output.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as target:
        target.write(serialized)
    return destination
