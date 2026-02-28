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

## Current State (as of 2026-02-28)

### Done
- Full project scaffold committed and pushed to GitHub: https://github.com/atrain291/fencing-analyzer
- Git identity: `atrain291` / `atrain291@gmail.com`
- Project lives on external drive: `/run/media/adeitz/63A504213DF71637/fencing-visualizer/`
- `git config core.fileMode false` set (NTFS drive — needed to suppress false file mode changes)

### Containers built successfully
- `localhost/fencing-visualizer_frontend:latest` (React + Vite + Tailwind)
- `localhost/fencing-visualizer_api:latest` (FastAPI)
- Postgres, Redis, Ollama images: pulled fine

### Worker image — NOT built yet
- **Blocker: `/var/home` partition is 97% full (4.3GB free)**
- `docker.io/pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime` needs ~8GB to unpack
- Podman stores image layers in `~/.local/share/containers/storage/` on `/var/home` (110GB SSD)
- Fix: redirect Podman storage to external drive (1.1TB free)

### Services — NOT running yet
- No containers started; need to resolve worker build first (or start without worker)

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
- **Home partition:** `/var/home` — `sda3`, 110GB, currently 97% full
- **External drive:** `/run/media/adeitz/63A504213DF71637/` — 1.9TB NTFS, ~1.1TB free
- **Podman compose:** `podman-compose` via linuxbrew at `/home/linuxbrew/.linuxbrew/bin/podman-compose`
- **GPU passthrough:** CDI-based (`devices: [nvidia.com/gpu=all]` in compose)
- **ANTHROPIC_API_KEY:** needs to be set in `.env` (not committed)

---

## Immediate Next Steps

### 1. Fix Podman storage location (unblocks worker build)
```bash
mkdir -p ~/.config/containers
cat > ~/.config/containers/storage.conf << 'EOF'
[storage]
driver = "overlay"
graphroot = "/run/media/adeitz/63A504213DF71637/podman-storage"
runroot = "/run/user/1000/containers"
EOF
# Then rebuild worker:
cd /run/media/adeitz/63A504213DF71637/fencing-visualizer
podman-compose build worker
```

### 2. Add Anthropic API key
```bash
# Edit .env — add your key:
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run database migrations
```bash
cd /run/media/adeitz/63A504213DF71637/fencing-visualizer
podman-compose up -d postgres
podman-compose run --rm api alembic upgrade head
```

### 4. Start all services
```bash
podman-compose up --build
```

### 5. Stage 1 completion — wire skeleton overlay
- Backend: `GET /api/bouts/{id}` needs to return frame keypoints
- Frontend: `VideoReview.tsx` canvas overlay needs to draw skeleton from keypoint data
- Worker: pose estimation writes to DB — frontend polls and renders

---

## Development Stages (from spec)

| Stage | Status | Description |
|---|---|---|
| 1 | 🔧 In progress | Video upload, pose estimation, skeleton overlay, Claude API feedback |
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
├── docker-compose.yml                  ← Podman-compatible, CDI GPU
├── .env                                ← secrets (not in git) — add ANTHROPIC_API_KEY
├── .env.example                        ← template
├── documents/
│   └── fencing_analyzer_architecture.docx  ← full spec (read this)
├── backend/
│   ├── app/main.py                     ← FastAPI entry point
│   ├── app/models/                     ← SQLAlchemy ORM models
│   ├── app/api/routes/                 ← upload.py, bouts.py, fencers.py
│   └── alembic/                        ← DB migrations
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

- **NTFS drive**: `git config core.fileMode false` already set; dotfiles work fine on this NTFS volume
- **Worker models/__init__.py**: has an incorrect import path (`from worker.app.models...`) — should be `from app.models...` — fix before running
- **Podman short names**: all Dockerfiles and compose use fully-qualified image names (`docker.io/...`) to avoid Bazzite's short-name resolution enforcement
- **No GPU CDI setup yet**: may need `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` before worker/ollama GPU containers work
