"""WHAM Celery tasks — dispatched by the main worker pipeline.

Reads 2D pose data from Postgres, runs WHAM 3D reconstruction per person,
writes MeshState rows back to Postgres.
"""
import logging

from app.celery_app import celery_app
from app.db import get_db_session
from app.models import Frame, MeshState

logger = logging.getLogger(__name__)


@celery_app.task(name="wham.tasks.run_mesh_reconstruction", bind=True)
def run_mesh_reconstruction(self, bout_id: int, video_path: str,
                            video_info: dict) -> dict:
    """Run WHAM 3D mesh reconstruction for both fencer and opponent.

    Called by the main worker pipeline after pose estimation completes.
    Reads frames from Postgres, runs WHAM per subject, writes MeshState rows.
    """
    from app.inference import run_wham_inference, release_model

    with get_db_session() as db:
        frames = (
            db.query(Frame)
            .filter(Frame.bout_id == bout_id)
            .order_by(Frame.timestamp_ms)
            .all()
        )

        if not frames:
            logger.warning("No frames found for bout %d", bout_id)
            return {"bout_id": bout_id, "status": "skipped", "reason": "no_frames"}

        logger.info("WHAM reconstruction for bout %d: %d frames", bout_id, len(frames))

        results = {}

        for subject, pose_key in [("fencer", "fencer_pose"), ("opponent", "opponent_pose")]:
            poses = []
            frame_ids = []
            for f in frames:
                pose = getattr(f, pose_key)
                poses.append(pose if pose else {})
                frame_ids.append(f.id)

            # Skip if subject has too few valid poses
            valid_count = sum(1 for p in poses if p)
            if valid_count < 10:
                logger.info("Skipping %s: only %d valid poses", subject, valid_count)
                results[subject] = {"status": "skipped", "reason": "insufficient_poses"}
                continue

            logger.info("Running WHAM for %s (%d valid poses)...", subject, valid_count)
            wham_result = run_wham_inference(video_path, poses, video_info)

            if wham_result is None:
                logger.warning("WHAM returned None for %s", subject)
                results[subject] = {"status": "failed"}
                continue

            # Write MeshState rows
            mesh_count = 0
            for t in range(min(wham_result["frame_count"], len(frame_ids))):
                mesh = MeshState(
                    frame_id=frame_ids[t],
                    subject=subject,
                    body_pose=wham_result["body_poses"][t],
                    global_orient=wham_result["global_orients"][t],
                    betas=wham_result["betas"],
                    joints_3d=wham_result["joints_3d"][t],
                    global_translation=None,  # requires homography correction
                    foot_contact=wham_result["foot_contacts"][t],
                    confidence=None,
                )
                db.add(mesh)
                mesh_count += 1

                if mesh_count % 500 == 0:
                    db.commit()

            db.commit()
            logger.info("Wrote %d MeshState rows for %s", mesh_count, subject)
            results[subject] = {"status": "complete", "frames": mesh_count}

        # Release WHAM model from GPU
        release_model()

    return {"bout_id": bout_id, "status": "complete", "results": results}
