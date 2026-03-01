# Claude Session Notes

Auto-updated by Claude Code approximately every 10 minutes.

---

## Session: 2026-03-01

### Project Status
- Stage 1 is **code complete**: video upload → YOLOv8-Pose skeleton overlay → Claude coaching feedback
- All Docker images built, DB migrated (9 tables), frontend (5173) + backend (8000) running
- **Worker container NOT yet started** — videos queue at 0% until worker is up

### Container Status (2026-03-01)
All 6 containers running: postgres (5432), redis (6379), ollama (11434), api (8000), frontend (5173), worker.

### Build Fix — SELinux RELRO Issue
Bazzite kernel blocks `mprotect()` in rootless Podman builds. Always build images with:
```bash
podman build --security-opt seccomp=unconfined --security-opt label=disable -t <name> <context-dir>/
```
docker-compose.yml updated to add `security_opt: [label=disable, seccomp=unconfined]` to frontend and api services.
Start containers with `podman-compose up -d --no-build` (images must be pre-built manually).

### .env File
Created from .env.example. `ANTHROPIC_API_KEY` is blank — must be filled in for LLM features.

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
