# Session 8 Handoff — 2026-03-05

## What Was Done

### Session 7 (2026-03-04)
- Fixed 4 housekeeping items: dashboard delete cascade, skeleton rendering artifacts, video muted default, corner-drag resize
- Filtered (0,0) keypoints in skeleton drawing for YOLO11x compatibility
- Commits: `21172db`, `13d1bc5`

### Session 8 (2026-03-05) — Priority 2 Assessment
- Evaluated readiness for Priority 2 blade detection refinements
- No code changes — assessment and planning only

## Current State
- **Branch**: `feature/stage-2-blade-detection`
- **Priority 1 COMPLETE** (commit `f9fa740`):
  - 1a. Persistent weapon arm — determined once from fencer orientation
  - 1b. EMA temporal smoothing (alpha=0.4) on tip position
  - 1c. Fixed blade length — calibrated from max arm extension

## Priority 2 Assessment Results

### 2a. Occlusion Bridging via Linear Interpolation
- **Status**: Not implemented
- **Scope**: ~100-150 lines in `blade.py`
- **What**: Buffer last N valid tip positions. If gap < 5 frames (~150ms at 30fps), linearly interpolate. If gap exceeds threshold, emit nothing and reset.
- **Key detail**: Currently, any keypoint confidence dropout resets ALL EMA state (smooth_x/y, prev values, prev_ts). Occlusion bridging replaces this hard reset with graceful interpolation.
- **No dependencies**

### 2b. Blade Visibility Confidence Score
- **Status**: Not implemented
- **Scope**: DB migration + ~80 lines across worker/backend/frontend
- **What**: Composite confidence from keypoint conf (40%), temporal consistency (40%), arm extension ratio (20%)
- **Blocking issue**: `BladeState` model has NO `confidence` column — needs:
  - New Alembic migration: `ALTER TABLE blade_states ADD COLUMN confidence FLOAT DEFAULT NULL`
  - Model update in both `worker/app/models/analysis.py` AND `backend/app/models/analysis.py`
  - Schema update in `backend/app/schemas/bout.py` (BladeStateRead)
  - Frontend type update in `frontend/src/api/bouts.ts`
- **Frontend rendering**: `drawBlade()` uses fixed `globalAlpha = 0.85` — change to `0.3 + 0.55 * confidence`
- **Frontend trail**: `drawTipTrail()` opacity gradient should multiply by confidence
- **Frontend interpolation**: `interpolateBladeState()` (line 175-186) should also interpolate confidence

### 2c. Action Classification Integration
- **Status**: Not implemented
- **Scope**: ~150-200 lines across actions.py, video_pipeline.py, blade.py
- **What**: Feed blade speed into lunge detection as supplementary signal; compute per-action blade metrics
- **Key issue**: Pipeline currently runs actions BEFORE blade tracking (actions determines orientation, blade uses it). Integration requires either:
  - Two-pass: extract orientation first, run blade, then run full action classification with blade data
  - Or split orientation detection out of actions.py into its own function
- **actions.py current state**: Purely ankle-based footwork detection (advance, retreat, lunge, step_lunge, check_step, recovery). Sliding window 400ms. Lunge = `dx_front > 0.45 AND abs(dx_back) < 0.15`.
- **Blade enhancement for lunges**: Real attack = foot extension + blade acceleration forward. Foot-only movement without blade = preparation, not attack.
- **Optional Action model extension**: `blade_speed_avg`, `blade_speed_peak`, `blade_linearity` columns
- **Depends on**: 2a (stable blade speed through occlusions)

## Recommended Implementation Order
1. **2a** (occlusion bridging) — foundational for stable blade speed
2. **2b** (confidence score) — quick win, improves visual feedback
3. **2c** (action integration) — highest user value, depends on 2a

## Key Files for Priority 2
| File | Role |
|------|------|
| `worker/app/pipeline/blade.py` | Core blade tracking — 2a and 2b changes here |
| `worker/app/pipeline/actions.py` | Action classification — 2c changes here |
| `worker/app/tasks/video_pipeline.py` | Pipeline orchestrator — 2c reordering here |
| `worker/app/models/analysis.py` | BladeState model — 2b adds confidence column |
| `backend/app/models/analysis.py` | Mirror BladeState model — 2b adds confidence column |
| `backend/app/schemas/bout.py` | API schema — 2b exposes confidence |
| `frontend/src/api/bouts.ts` | Frontend types — 2b adds confidence field |
| `frontend/src/pages/VideoReview.tsx` | Blade rendering — 2b opacity modulation |

## Previous Session Handoff
See git history for session 6 notes (commit `bd6d878`).
