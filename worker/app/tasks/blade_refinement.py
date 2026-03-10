"""Celery task for WHAM 3D blade refinement.

Dispatched by the WHAM worker after mesh_states are committed.
Reads MeshState + BladeState from DB, refines blade tip positions
using 3D wrist rotation and wrist→hand direction vectors.
"""
import logging

from app.celery_app import celery_app
from app.db import get_db_session

logger = logging.getLogger(__name__)


@celery_app.task(name="worker.tasks.blade_refinement.refine_blade_with_mesh", bind=True)
def refine_blade_with_mesh(self, bout_id: int, video_info: dict,
                           orientation: str | None = None) -> dict:
    """Refine blade tracking with WHAM 3D data (Pass 2).

    Called by WHAM worker after mesh_states are written to DB.
    """
    from app.pipeline.blade_refinement import refine_blade_from_mesh

    logger.info("Starting blade refinement for bout %d", bout_id)

    with get_db_session() as db:
        try:
            refined = refine_blade_from_mesh(bout_id, db, video_info,
                                             orientation=orientation)
            logger.info("Blade refinement complete for bout %d: %d frames", bout_id, refined)
            return {"bout_id": bout_id, "status": "complete", "refined_frames": refined}
        except Exception:
            logger.exception("Blade refinement failed for bout %d", bout_id)
            return {"bout_id": bout_id, "status": "failed"}
