"""
Video analysis pipeline — Stage 2.

Pipeline stages (per architecture doc):
  1.  Video ingest & decode          FFmpeg + NVDEC
  2.  Pose estimation per frame      RTMPose WholeBody (CUDA)
  3.  Guard / shield detection       Custom model (CUDA)        [Stage 2 — heuristic for now]
  4.  Blade axis inference           Geometric calc             [Stage 2]
  5.  Tip position (nominal)         Projection                 [Stage 2]
  6.  Depth estimation               Depth Anything V2          [Stage 3]
  7.  Blade flex simulation          PyTorch physics            [Stage 3]
  8.  3D tip trajectory              PyTorch tensor ops         [Stage 3]
  9.  Opponent distance & velocity   Vector math                [Stage 4]
  10. Correction cost model          Angular calc               [Stage 4]
  11. Kinetic chain readiness        Joint angle analysis       [Stage 3]
  12. Off-hand analysis              Pose keypoints             [Stage 3]
  13. Action classification          Rule-based + ML            [Stage 2]
  14. Technique scoring              Reference comparison       [Stage 5]
  15. LLM synthesis                  Llama + Claude             [Stage 1]
  16. Practice plan generation       Claude API                 [Stage 5]
"""
import os
import logging
from celery import current_task

from app.celery_app import celery_app
from app.db import get_db_session
from app.pipeline.ingest import ingest_video
from app.pipeline.pose import run_pose_estimation
from app.pipeline.blade import run_blade_tracking
from app.pipeline.actions import run_action_classification
from app.pipeline.orientation import detect_orientation
from app.pipeline.llm import synthesize_coaching_feedback

logger = logging.getLogger(__name__)


def _release_gpu():
    """Release RTMPose model from GPU memory between pipeline stages."""
    import gc
    import torch
    from app.pipeline import pose
    pose._model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("GPU memory released (RTMPose unloaded)")


def _dispatch_wham(bout_id: int, video_path: str, video_info: dict,
                   orientation: str | None = None):
    """Dispatch WHAM 3D reconstruction to the wham worker queue.

    This is fire-and-forget: the main pipeline continues with blade tracking
    while WHAM runs asynchronously in a separate container. Results are
    written directly to the mesh_states table by the WHAM worker.
    After WHAM completes, it dispatches a blade refinement callback.
    """
    from celery import Celery
    wham_app = Celery(broker=os.environ.get("REDIS_URL", "redis://redis:6379/0"))
    wham_app.send_task(
        "wham.tasks.run_mesh_reconstruction",
        args=[bout_id, video_path, video_info, orientation],
        queue="wham_pipeline",
    )


STAGES = [
    "ingest",
    "pose_estimation",
    "mesh_reconstruction",  # async — WHAM in separate container
    "blade_tracking",
    "action_classification",
    # "llm_synthesis",  # disabled — re-enable when LLM coaching is needed
]


def _update_progress(bout_id: int, stage: str, pct: int, db, extra: dict | None = None):
    import psutil
    import torch

    metrics = {}
    try:
        metrics["cpu_pct"] = psutil.cpu_percent(interval=None)
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            metrics["gpu_mem_pct"] = round(allocated / total * 100, 1)
    except Exception:
        pass

    from app.models import Bout
    bout = db.query(Bout).get(bout_id)

    # Merge with existing pipeline_progress so fields from earlier stages
    # (e.g. total_frames set during ingest) are preserved across updates.
    existing = {}
    if bout and isinstance(bout.pipeline_progress, dict):
        existing = dict(bout.pipeline_progress)

    progress = {**existing, "stage": stage, "pct": pct, **metrics, **(extra or {})}

    if bout:
        bout.pipeline_progress = progress
        db.commit()
    current_task.update_state(state="PROGRESS", meta=progress)


@celery_app.task(name="worker.tasks.video_pipeline.run_pipeline", bind=True)
def run_pipeline(self, bout_id: int, video_path: str):
    logger.info("Starting pipeline for bout %d, video: %s", bout_id, video_path)

    with get_db_session() as db:
        from app.models import Bout
        from app.models.analysis import Frame

        bout = db.query(Bout).get(bout_id)
        if not bout:
            raise ValueError(f"Bout {bout_id} not found")
        bout.status = "processing"
        db.commit()

        try:
            # Stage 1 — Ingest
            _update_progress(bout_id, "ingest", 5, db)
            video_info = ingest_video(video_path)
            logger.info("Ingest complete: %s", video_info)

            # Stage 2 — Pose estimation (20→80%)
            _update_progress(bout_id, "pose_estimation", 20, db,
                             extra={"frame": 0, "total_frames": video_info.get("total_frames", 0)})

            def pose_progress(frame_idx: int, total_frames: int):
                if total_frames > 0:
                    frame_pct = frame_idx / total_frames
                    pct = int(20 + frame_pct * 60)  # scale 20→80
                else:
                    pct = 20
                _update_progress(bout_id, "pose_estimation", pct, db,
                                 extra={"frame": frame_idx, "total_frames": total_frames})

            fencer_bbox = bout.fencer_bbox  # dict or None
            opponent_bbox = bout.opponent_bbox  # dict or None
            strip = (bout.preview_data or {}).get("piste")  # strip polygon or None

            pose_results = run_pose_estimation(video_path, video_info, bout_id, db,
                                               progress_callback=pose_progress,
                                               fencer_bbox=fencer_bbox,
                                               opponent_bbox=opponent_bbox,
                                               strip=strip)
            logger.info("Pose estimation complete: %d frames", len(pose_results))

            # Release RTMPose from GPU before downstream stages
            _release_gpu()

            # Load frame ORM objects for downstream stages
            frames = (db.query(Frame)
                      .filter(Frame.bout_id == bout_id)
                      .order_by(Frame.timestamp_ms)
                      .all())

            # Detect fencer orientation (needed by both WHAM refinement and blade tracking)
            orientation = detect_orientation(frames)

            # Stage 2.5 — WHAM 3D mesh reconstruction (async, separate container)
            # Orientation passed through so WHAM can dispatch blade refinement callback
            _update_progress(bout_id, "mesh_reconstruction", 81, db)
            try:
                _dispatch_wham(bout_id, video_path, video_info, orientation=orientation)
                logger.info("WHAM task dispatched for bout %d", bout_id)
            except Exception:
                logger.warning("WHAM dispatch failed — continuing without 3D data", exc_info=True)

            # Stage 3 — Blade tracking (Pass 1: 2D keypoints only)
            _update_progress(bout_id, "blade_tracking", 83, db)
            run_blade_tracking(frames, video_info, db, orientation=orientation)
            logger.info("Blade tracking complete")

            # Stage 4 — Action classification (uses blade speed data)
            _update_progress(bout_id, "action_classification", 86, db)
            from app.models.analysis import BladeState
            blade_rows = (db.query(BladeState)
                          .join(Frame, BladeState.frame_id == Frame.id)
                          .filter(Frame.bout_id == bout_id)
                          .all())
            fencer_blade_speeds = {
                bs.frame_id: {"speed": bs.speed, "confidence": bs.confidence}
                for bs in blade_rows if bs.subject == "fencer"
            }
            opponent_blade_speeds = {
                bs.frame_id: {"speed": bs.speed, "confidence": bs.confidence}
                for bs in blade_rows if bs.subject == "opponent"
            }
            action_results = run_action_classification(bout_id, frames, db,
                                                       blade_speeds=fencer_blade_speeds,
                                                       opponent_blade_speeds=opponent_blade_speeds)
            logger.info("Action classification complete: %d actions", len(action_results))

            # Stage 5 — LLM synthesis (DISABLED — skip to complete)
            # To re-enable, uncomment the block below and remove the stub.
            logger.info("LLM synthesis stage disabled — skipping for bout %d", bout_id)
            coaching_text = "LLM coaching disabled. Re-enable llm_synthesis stage in video_pipeline.py."
            # _update_progress(bout_id, "llm_synthesis", 90, db)
            #
            # # Collect blade state data for LLM prompt enrichment
            # from app.models.analysis import BladeState
            # blade_rows = (db.query(BladeState)
            #               .join(Frame, BladeState.frame_id == Frame.id)
            #               .filter(Frame.bout_id == bout_id)
            #               .order_by(Frame.timestamp_ms)
            #               .all())
            # blade_states = [
            #     {"speed": bs.speed, "tip_xyz": bs.tip_xyz, "velocity_xyz": bs.velocity_xyz}
            #     for bs in blade_rows
            # ]
            # logger.info("Loaded %d blade states for LLM synthesis", len(blade_states))
            #
            # coaching_text = synthesize_coaching_feedback(
            #     bout_id, pose_results, action_results, blade_states, db
            # )

            # Persist analysis (upsert — old row may exist from a previous run)
            from app.models import Analysis
            existing = db.query(Analysis).filter(Analysis.bout_id == bout_id).first()
            if existing:
                existing.technique_scores = {}
                existing.patterns = {"actions": action_results}
                existing.fatigue_markers = {}
                existing.llm_summary = coaching_text
            else:
                db.add(Analysis(
                    bout_id=bout_id,
                    technique_scores={},
                    patterns={"actions": action_results},
                    fatigue_markers={},
                    llm_summary=coaching_text,
                ))
            bout.status = "complete"
            bout.pipeline_progress = {"stage": "complete", "pct": 100}
            db.commit()

            logger.info("Pipeline complete for bout %d", bout_id)
            return {"bout_id": bout_id, "status": "complete"}

        except Exception as exc:
            logger.exception("Pipeline failed for bout %d", bout_id)
            bout.status = "failed"
            bout.error = str(exc)
            db.commit()
            raise
