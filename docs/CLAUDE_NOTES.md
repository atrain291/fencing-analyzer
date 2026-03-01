# Claude Session Notes

Auto-updated by Claude Code approximately every 10 minutes.

---

## Session: 2026-03-01

### Project Status
- Stage 1 is **code complete**: video upload → YOLOv8-Pose skeleton overlay → Claude coaching feedback
- All Docker images built, DB migrated (9 tables), frontend (5173) + backend (8000) running
- **Worker container NOT yet started** — videos queue at 0% until worker is up

### Immediate Next Step
Start the Celery worker from the external drive directory:
```bash
cd /run/media/adeitz/63A504213DF71637/fencing-visualizer
podman-compose up -d worker
podman logs -f fencing-visualizer_worker_1
```

### Key Architecture Reminder
- **Source repo**: `/var/home/adeitz/source/fencing-analyzer/`
- **Runtime (podman-compose)**: `/run/media/adeitz/63A504213DF71637/fencing-visualizer/`
- Pipeline: ingest (5%) → pose estimation (20%) → LLM synthesis (85%) → complete (100%)
- SELinux `label=disable` required on bind mounts
- GPU: CDI passthrough `nvidia.com/gpu=all`

### Stages Remaining After Stage 1
2. Guard/blade detection, tip trajectory, action classification
3. Depth estimation, 3D blade tracking, flex physics
4. Threat analysis, correction cost, cone of commitment
5. Reference skeletons, technique comparison, practice plans
6. Longitudinal tracking, fatigue detection
7. Live camera feed
