# Session 6 Handoff — 2026-03-04

## What Was Done

### 1. H.264 Transcode Task (new)
- **File**: `worker/app/tasks/transcode.py`
- Auto-dispatched on upload alongside preview task
- HEVC → H.264 via NVENC GPU (~2s for 30MB video), falls back to libx264
- Updates `bouts.video_url` to point to `_web.mp4` file
- **Wiring**: `backend/app/tasks.py` (`dispatch_transcode`), `backend/app/api/routes/upload.py` (auto-dispatch), `worker/app/celery_app.py` (include)

### 2. API Memory Leak Fix (critical)
- **Root cause**: `GET /api/bouts/{id}` eagerly loaded ALL frames (367+) with pose JSON + blade states via `joinedload()`. Each response was massive, and uvicorn holding connections caused 18GB+ memory usage.
- **Fix**: Split into two endpoints:
  - `GET /api/bouts/{id}` — metadata + analysis only (lightweight)
  - `GET /api/bouts/{id}/frames` — all frames + blade states + actions (heavy, called once by VideoReview)
- **Files changed**: `backend/app/api/routes/bouts.py`, `backend/app/schemas/bout.py`, `frontend/src/api/bouts.ts`, `frontend/src/pages/VideoReview.tsx`

### 3. Worker Model Sync Fixes
- `worker/app/models/bout.py` — added missing `video_url` column
- `worker/app/models/analysis.py` — added missing `created_at` column with `server_default=func.now()`
- **Note**: Worker and backend have SEPARATE model files. They must be kept in sync.

### 4. DB Schema Fix
- `analyses.created_at` column exists in Alembic migrations but has no DEFAULT value
- **Manual fix required after fresh DB**: `ALTER TABLE analyses ALTER COLUMN created_at SET DEFAULT now();`
- Should be added to Alembic migration or init script for permanence

### 5. Clean Slate
- DB was wiped (all fencers, bouts, uploads removed)
- Fresh postgres init + `alembic upgrade head` + manual ALTER TABLE
- 1 bout (bout #1) processed end-to-end: upload → preview → transcode → skeleton selection → pipeline → review
- Verified: video plays in Firefox, skeletons render, action timeline works, API stable at ~100MB

## Commits This Session
```
60ca600 Fix API memory leak and Firefox video playback
3248aa9 Add H.264 transcode task for Firefox playback, add CLAUDE.md
```

## Current State
- **Branch**: `feature/stage-2-blade-detection`
- **All services running**: 6 containers (frontend, api, worker, postgres, redis, ollama)
- **API memory**: ~100MB (was 18GB+)
- **DB**: Fresh, 1 bout processed, `analyses.created_at` DEFAULT manually set
- **Images**: All 3 rebuilt from scratch this session

## Known Remaining Issues
1. **`analyses.created_at` DEFAULT** not in Alembic migrations — needs a new migration or the ALTER TABLE must be run manually after every fresh DB init
2. **Playwright MCP** uses Firefox (`--browser firefox` in `.mcp.json`)
3. **`test_video.mp4`** in project root is a temp file (can be deleted)

## What's Next
- **P1 Blade Detection Refinement** (see `documents/blade_detection_refinement_plan.md`):
  - Weapon arm persistence (detect which arm holds weapon)
  - EMA smoothing on tip position
  - Fixed blade length constraint
- These are the next items for Stage 2 completion
