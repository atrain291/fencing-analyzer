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

### Worker Restart Protocol
Always clear Python bytecode cache before restarting or code changes won't load:
```bash
podman exec fencing-analyzer_worker_1 find /app -name "*.pyc" -delete
podman restart fencing-analyzer_worker_1
```
If a bout was in-progress when worker restarted, reset it manually:
```bash
# In postgres container
DELETE FROM frames WHERE bout_id = <id>;
UPDATE bouts SET status='queued', pipeline_progress='{"stage":"queued","pct":0}', task_id=NULL WHERE id=<id>;
# In api container
python3 -c "from app.tasks import dispatch_pipeline; t=dispatch_pipeline(<id>,'/app/uploads/<key>.mp4'); print(t.id)"
# Then update task_id in DB
```

---

## Session 2 Summary — 2026-03-01

### Features Completed
- Cancel upload mid-flight + cancel/delete processing bout
- Skeleton overlay: rAF loop, binary search frame lookup, keypoint interpolation
- Granular pipeline progress: weighted stage bars, frame counter, GPU/CPU bars, ETA estimate
- Total frames reported from ingest stage immediately (nb_frames from ffprobe)
- NVDEC hardware video decoding (ffmpeg pipe, hevc_cuvid/h264_cuvid auto-selection)
- Slow motion playback buttons (0.25×, 0.5×, 1×, 2×)
- Wider video layout + fullscreen with canvas overlay preserved (ResizeObserver)
- SAM-Body4D added to architecture doc as Section 17

### Known Gotchas
- `ffprobe -count_packets` broken in ffmpeg 4.4 — using `nb_frames` instead
- Video file for bout #12 (4K HEVC, 9753 frames) was lost from external drive — deleted from DB
- Bout #17 was processed successfully with 1080p 30fps video (12s, hevc)

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
