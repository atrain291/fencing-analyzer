# Session 14 Handoff — 2026-03-10

## What Was Done

### Dual-Subject Blade Tracking (COMPLETE)
- **BladeState now tracks both fencer and opponent**: added `subject` column ("fencer"/"opponent")
- Removed `unique=True` constraint on `frame_id` — one row per subject per frame
- Refactored `blade.py`: `run_blade_tracking()` calls `_run_blade_for_subject()` for both subjects
- Alembic migration `b9c0d1e2f3a4`: adds subject column, drops unique frame_id constraint
- Backend model/schema/route updated: `Frame.blade_states` (plural list), serializes by subject
- Frontend renders opponent blade in cyan (`#06b6d4`), fencer blade in green (`#22c55e`)

### Forward Direction Fix — Position-Based (COMPLETE)
- **Root cause**: Orientation detection (`detect_orientation()`) was inverted for side-profile views
  - Used left/right shoulder x-positions, but in COCO the anatomical right shoulder is always
    more to the right regardless of facing direction in near-profile views
  - This cascaded to wrong weapon arm detection AND wrong forward-hemisphere clamp
  - Both blades pointed away from each other instead of toward each other
- **Fix**: New `_compute_forward_sign()` function computes forward direction from actual
  hip-center positions of both fencers, bypassing the broken orientation heuristic entirely
  - Returns +1.0 if opponent is to the right, -1.0 if to the left
  - Used for both weapon arm detection and forward-hemisphere clamp
  - `_determine_weapon_arm_from_poses()` now takes `forward_sign` instead of `orientation`
  - `_run_blade_for_subject()` derives `subject_orientation` from `forward_sign` for angulation
- Verified: fencer tip_x=0.565 (right of wrist 0.478, toward opponent), opponent tip_x=0.591 (left of wrist 0.679, toward fencer)

### Two-Pass Blade Refinement Architecture (COMPLETE — from session 13)
- Pass 1: 2D keypoints only (immediate, in main pipeline)
- Pass 2: WHAM 3D refinement (async callback after mesh reconstruction)
- `blade_refinement.py`: uses SMPL wrist rotation + 3D wrist-hand vector
- Celery task dispatched by WHAM worker after mesh_states committed

### Hand Keypoint Direction Improvements (from session 13)
- Replaced fine-grained MCP-PIP (3-4px, too noisy) with wrist-fingertip-centroid (~20px)
- Minimum span gate: `_HAND_MIN_SPAN_NORM = 0.008` (~15px at 1920w)
- Hand direction capped at 50% blend weight, forearm always contributes
- Angulation range 15-45 deg (was 5-25), inverted mapping (bent arm = max deflection)

### Fullscreen Overlay Fixes (from session 13)
- `playerPanelRef` fullscreens entire left column (video + scrubber + timeline + controls)
- Canvas dynamically positioned for `object-contain` letterboxing
- Hidden native fullscreen button, redirect fallback

## Current State
- **Branch**: `feature/stage-3-blade-refinement`
- **All 7 containers running**: frontend(:5174), api(:8001), worker(GPU), wham(GPU), postgres(:5433), redis(:6380), ollama(:11435)
- **Blade tracking**: dual-subject (fencer + opponent), position-based forward direction
- **Migrations applied**: through `b9c0d1e2f3a4` (blade_states subject column)
- **Migrations pending**: `c4d5e6f7a8b9` (confidence), `d5e6f7a8b9c0` (blade speed), `e6f7a8b9c0d1` (unique fencer name)

## Known Issues / Next Steps for Blade Tracking
- Blade direction still needs refinement — correct general direction but accuracy varies
- Possible improvements:
  - Use wrist-to-fingertip vector more aggressively when hand keypoints are available
  - Per-frame opponent position for forward direction (currently uses average across frames)
  - Leverage WHAM 3D wrist rotation data when available (Pass 2 refinement)
  - Temporal smoothing improvements — Kalman filter may over-smooth direction changes
  - Investigate pixel-based blade detection (Hough lines, edge detection in hand region)

## Branch Lineage
```
master
  └─ feature/stage-2-blade-detection (P1, P2a, P2b)
       └─ feature/stage-2c-action-blade-integration (P2c)
            └─ feature/stage-3-blade-refinement (P3a, P3b, Strip, Housekeeping, RTMPose)  <- current
```

## What's Next

### WHAM Downstream Consumers (HIGH PRIORITY)
- Populate `kinetic_states` from WHAM 3D joints (joint angles, angular velocities)
- Populate `threat_metrics` from kinetic states (distance-to-target, attack tempo)
- These tables exist but have no writers yet

### Strip Homography
- `pipeline/homography.py` for strip-based world coordinate correction
- Replaces WHAM's unreliable global trajectory for fencing

### Frontend Updates
- 3D mesh visualization (consume WHAM mesh_states data)
- Toggle hand skeleton rendering in review UI (infrastructure ready, `showHands` param)
- Display `blade_speed_avg` / `blade_speed_peak` per action in review UI

### Future Research
- 4D Gaussian splatting for multi-camera fencing replay
- LLM coaching re-enablement (Claude API integration exists, currently stubbed)
- Pixel-based blade detection (complement geometric projection)

## Commits This Session
```
(pending) Fix blade direction: position-based forward direction, dual-subject tracking
c04cd20 Add WHAM 3D blade refinement pass (two-pass architecture)
03cc680 Update handoff for session 13
fa09687 Fix skeleton overlay disappearing on fullscreen maximize
```

## Previous Sessions
- Session 13: WHAM 3D blade refinement, hand keypoint improvements, fullscreen overlay fix
- Session 12: WHAM end-to-end, hand keypoints, fullscreen fix, foot keypoints
- Session 11: WHAM container setup, foot keypoints started, per-fencer action timeline
- Session 10: RTMPose migration, GPU fix, configurable ports, self-contained project
- Session 9: P2c action-blade integration, P3a wrist angulation, P3b Kalman filter, strip detection, fencer housekeeping
- Session 8: P2a occlusion bridging + P2b confidence score (`296b586`)
