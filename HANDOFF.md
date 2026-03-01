# Fencing Analyzer — Session Handoff

> Read this file at the start of every new Claude Code session to restore full context.

---

## What This Project Is

A web-based AI coaching platform for **epee fencing**. Fencers upload bout footage and receive:
- Pose estimation skeleton overlay (YOLOv8-Pose)
- Epee tip tracking via physics modeling
- Kinetic chain / biomechanical analysis
- Natural language coaching feedback via Claude API + local Llama

Full spec: `documents/fencing_analyzer_architecture.docx`

---

## Current State (as of 2026-03-01, session 2)

### Done
- Full project scaffold committed and pushed to GitHub: https://github.com/atrain291/fencing-analyzer
- Git identity: `atrain291` / `atrain291@gmail.com`
- Project lives on external drive: `/run/media/adeitz/63A504213DF71637/fencing-visualizer/`
- `git config core.fileMode false` set (NTFS drive — needed to suppress false file mode changes)
- **Stage 1 skeleton overlay feature — code complete and committed**
- **Podman storage redirected to external drive** (`~/.config/containers/storage.conf`)
- **All images built** (frontend, api, worker, postgres, redis pulled)
- **DB migrations generated and applied** — 9 tables created
- **Frontend running** via `npm run dev` on port 5173
- **postgres, redis, api containers running**

### What was happening when session ended
- User uploaded a bout video and it was stuck at "queued / 0%"
- Root cause: **Celery worker container was not started**
- `podman-compose up -d worker` was just issued but interrupted

### IMMEDIATE NEXT STEP: Start the worker
```bash
cd /run/media/adeitz/63A504213DF71637/fencing-visualizer
podman-compose up -d worker
```
Then watch logs to confirm it picks up the queued job:
```bash
podman logs -f fencing-visualizer_worker_1
```

---

## How to Restore the Full Dev Environment

### 1. Start containers (postgres, redis, api, worker)
```bash
cd /run/media/adeitz/63A504213DF71637/fencing-visualizer
podman-compose up -d postgres redis api worker
```

### 2. Start frontend dev server
```bash
cd /run/media/adeitz/63A504213DF71637/fencing-visualizer/frontend
npm run dev
```
Frontend: http://localhost:5173 (or http://192.168.1.197:5173 over network)
API: http://localhost:8000

### 3. Verify everything is up
```bash
podman ps
```
Should show: postgres_1, redis_1, api_1, worker_1

---

## Podman / Storage Fix (already applied — for reference)

- Podman image storage redirected to external drive: `~/.config/containers/storage.conf`
  - `graphroot = "/run/media/adeitz/63A504213DF71637/podman-storage"`
- **Named volumes for postgres/redis were broken** (NTFS + SELinux + rootless UID mapping issues)
- **Fix applied in docker-compose.yml**: switched to bind mounts on `/var/home/adeitz/fencing-data/`
  - postgres data: `/var/home/adeitz/fencing-data/postgres`
  - redis data: `/var/home/adeitz/fencing-data/redis`
- Both postgres and redis have `security_opt: label=disable` (SELinux fix for Bazzite)
- Bind mount dirs already created with correct ownership — **do not delete them**

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + TypeScript + Vite + Tailwind CSS |
| Backend API | FastAPI + SQLAlchemy + Alembic |
| Job Queue | Celery + Redis |
| ML Worker | PyTorch CUDA + YOLOv8-Pose + OpenCV + FFmpeg |
| Local LLM | Ollama + Llama 3.1 8B |
| Cloud LLM | Claude API (Anthropic) |
| Database | PostgreSQL |
| Storage | Local uploads (S3/R2 later) |
| Containers | Podman + podman-compose |
| Host OS | Bazzite (immutable) |
| GPU | Nvidia 4070 Super |

---

## Environment Details

- **OS:** Bazzite (Fedora immutable) on `sda` (110GB)
- **Home partition:** `/var/home` — `sda3`, 110GB, ~25GB free
- **External drive:** `/run/media/adeitz/63A504213DF71637/` — 1.9TB NTFS, ~1.1TB free
- **Podman compose:** `podman-compose` via linuxbrew at `/home/linuxbrew/.linuxbrew/bin/podman-compose`
- **GPU passthrough:** CDI-based (`devices: [nvidia.com/gpu=all]` in compose)
- **ANTHROPIC_API_KEY:** needs to be set in `.env` (not committed) — check if set before testing LLM features

---

## Development Stages (from spec)

| Stage | Status | Description |
|---|---|---|
| 1 | ✅ Code complete — worker not yet tested end-to-end | Video upload, pose estimation, skeleton overlay, Claude API feedback |
| 2 | ⬜ Pending | Guard detection, blade axis, 2D tip trajectory |
| 3 | ⬜ Pending | Depth estimation, 3D tip, blade flex physics |
| 4 | ⬜ Pending | Threat analysis, correction cost, cone of commitment |
| 5 | ⬜ Pending | Reference skeletons, technique comparison, practice plans |
| 6 | ⬜ Pending | Longitudinal tracking, fatigue detection |
| 7 | ⬜ Pending | Live camera feed |

---

## Key File Locations

```
fencing-visualizer/
├── HANDOFF.md                          ← you are here
├── docker-compose.yml                  ← Podman-compatible, CDI GPU, bind mounts for pg/redis
├── .env                                ← secrets (not in git) — add ANTHROPIC_API_KEY
├── .env.example                        ← template
├── documents/
│   └── fencing_analyzer_architecture.docx  ← full spec (read this)
├── backend/
│   ├── app/main.py                     ← FastAPI entry point
│   ├── app/models/                     ← SQLAlchemy ORM models
│   ├── app/api/routes/                 ← upload.py, bouts.py, fencers.py
│   └── alembic/versions/               ← initial migration generated and applied
├── worker/
│   ├── app/tasks/video_pipeline.py     ← main pipeline orchestrator
│   └── app/pipeline/
│       ├── ingest.py                   ← FFmpeg metadata extraction
│       ├── pose.py                     ← YOLOv8-Pose per-frame
│       └── llm.py                      ← Claude API coaching synthesis
└── frontend/
    └── src/
        ├── pages/Dashboard.tsx         ← upload UI
        ├── pages/ProcessingStatus.tsx  ← polling progress
        └── pages/VideoReview.tsx       ← video + skeleton canvas
```

---

## Known Issues / Notes

- **SELinux on Bazzite**: all containers that use bind mounts need `security_opt: label=disable`
- **Rootless Podman + bind mounts**: named volumes had UID-mapping issues; switched to bind mounts on `/var/home/adeitz/fencing-data/` which is ext4 and supports POSIX permissions
- **Bind mount dirs**: `/var/home/adeitz/fencing-data/postgres` and `/var/home/adeitz/fencing-data/redis` — owned correctly, do not delete
- **GPU CDI**: may need `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` if worker/ollama can't see GPU
- **Ollama not started**: not needed until LLM coaching feedback is tested
- **NTFS drive**: `git config core.fileMode false` already set
