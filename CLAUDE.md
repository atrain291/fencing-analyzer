# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Web-based AI coaching platform for **epee fencing**. Fencers upload bout video and get pose estimation skeleton overlays (YOLO11x-Pose via ultralytics), blade tip tracking, action classification, and drill reports. Claude API coaching is implemented but currently disabled in the pipeline.

## Build & Run Commands

All services run in Podman containers via podman-compose.

```bash
# Build images (all three require these security opts for Bazzite/SELinux)
podman build --security-opt seccomp=unconfined --security-opt label=disable -t fencing-analyzer-api backend/
podman build --security-opt seccomp=unconfined --security-opt label=disable -t fencing-analyzer-frontend frontend/
podman build --security-opt seccomp=unconfined --security-opt label=disable -t fencing-analyzer-worker worker/

# Start all services (6 containers: frontend, api, worker, postgres, redis, ollama)
podman-compose up -d --no-build

# Full restart (required after code changes to worker — kills in-progress tasks)
podman-compose down && podman-compose up -d --no-build

# Database migrations (run inside api container or host with psycopg2)
cd backend && alembic upgrade head

# Frontend lint
cd frontend && npm run lint

# View logs
podman logs fencing-analyzer_worker_1   # ML pipeline logs
podman logs fencing-analyzer_api_1      # API logs
```

There is no test suite configured. Validation is done via manual E2E testing and Playwright MCP.

## Architecture

**Three-tier containerized system:**

```
Frontend (React 18 + Vite, :5173)
    ↓ /api proxy (vite.config.ts, 600s timeout)
Backend API (FastAPI, :8000)
    ↓ Celery dispatch
Worker (PyTorch + YOLO + FFmpeg, GPU)
    ↓ reads/writes
PostgreSQL 16 + Redis 7
```

### Frontend (`frontend/src/`)
React 18 + TypeScript + Tailwind CSS. Five routes in `App.tsx`:
- `/dashboard` — fencer CRUD, video upload, bout library
- `/bouts/:boutId/configure` — skeleton selection from preview frames
- `/bouts/:boutId/processing` — real-time progress polling
- `/bouts/:boutId/review` — video + canvas skeleton overlay + blade trail
- `/bouts/:boutId/drill` — rhythm/tempo/consistency scores

API client in `api/` uses Axios with `baseURL: /api`. Upload has `timeout: 0` for large files.

### Backend API (`backend/app/`)
FastAPI with SQLAlchemy ORM + Alembic migrations (3 applied). Key routes:
- `api/routes/upload.py` — chunked 1MB upload, auto-dispatches preview task
- `api/routes/bouts.py` — ROI config, drill report, status polling, CRUD
- `tasks.py` — Celery shim (`dispatch_pipeline`, `dispatch_preview`)

### ML Worker (`worker/app/`)
Celery worker processing one video at a time (`worker_prefetch_multiplier=1`).

**Pipeline stages** (orchestrated by `tasks/video_pipeline.py`):
1. **Ingest** — FFprobe metadata extraction
2. **Pose Estimation** (`pipeline/pose.py`) — YOLO11x-Pose (`yolo11x-pose.pt`) + BoT-SORT tracking with custom ID locking/re-lock logic. Hardware-decoded frames via FFmpeg NVDEC pipe.
3. **Blade Tracking** (`pipeline/blade.py`) — geometric tip projection from wrist/elbow keypoints
4. **Action Classification** (`pipeline/actions.py`) — rule-based footwork detection (advance, retreat, lunge, etc.) from ankle velocity thresholds
5. **LLM Synthesis** (`pipeline/llm.py`) — Claude API coaching (currently disabled, returns stub)

**Preview task** (`tasks/preview.py`): extracts 5 frames with YOLO predict (not track) for skeleton selection UI.

### Bout Status Flow
`configuring` → `previewing` → `preview_ready` → `queued` → `processing` → `complete` / `failed`

### Database (9 tables)
Core: `fencers`, `sessions`, `bouts` (with `preview_data`, `fencer_bbox`, `opponent_bbox`, `pipeline_progress` JSON columns), `frames` (with `fencer_pose`/`opponent_pose` JSON), `actions`, `blade_states`, `analyses`. Also `threat_metrics` and `kinetic_states` (for future stages).

## Key Architectural Details

**Tracker ID locking** (`pose.py`): Initial lock uses YOLO-precise bbox from preview to match tracker ID on first frame. Re-lock after occlusion uses proximity-only search with 3-frame confirmation gate. Tracker IDs don't persist across YOLO sessions — preview and pipeline use separate model instances.

**Custom BoT-SORT config** (`botsort_fencing.yaml`): `track_buffer=90` (3s occlusion resilience), `gmc_method: none` (fixed camera), `with_reid: False` (broken in ultralytics 8.3.0).

**Pose frame 0 fix**: First frame uses `persist=False` to avoid stale BoT-SORT/GMC state (OpenCV lkpyramid error). Preview task must clear tracker callbacks before `model.predict()`.

## Environment & Platform Gotchas

- **Host**: Bazzite (Fedora immutable) with Nvidia 4070 Super
- **SELinux**: All bind mounts need `label=disable` (already set in docker-compose.yml)
- **GPU passthrough**: CDI (`nvidia.com/gpu=all`). May require `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`
- **DB credentials**: user=`fencing`, db=`fencing_analyzer`, password=`fencing`
- **`VITE_API_URL`**: Must be `http://api:8000` in docker-compose (not localhost)
- **FFmpeg**: `ffprobe -count_packets` broken in ffmpeg 4.4 — use `nb_frames` from `-show_streams`
- **Worker restart**: kills in-progress tasks. Must reset bout status and re-dispatch manually.
- **Upload timeouts**: frontend `timeout: 0`, vite proxy 600s, uvicorn `--timeout-keep-alive 120`
- **TypeScript**: strict mode enabled, path alias `@/*` → `src/*`

## Development Stages

| Stage | Status | Focus |
|-------|--------|-------|
| 1 | Complete | Upload, skeleton overlay, Claude coaching |
| 2 | In progress | Blade detection refinement, action classification, skeleton selection UI |
| 3-7 | Planned | Depth estimation, threat analysis, reference skeletons, longitudinal tracking, live camera |

Current blade detection refinement priorities are documented in `documents/blade_detection_refinement_plan.md`.
