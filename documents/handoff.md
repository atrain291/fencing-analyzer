# Session 14 Handoff — 2026-03-11

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
  - COCO left/right shoulder positions don't reliably indicate facing direction in profile views
  - Cascaded to wrong weapon arm detection AND wrong forward-hemisphere clamp
- **Fix**: New `_compute_forward_sign()` — computes forward direction from actual hip positions
  of both fencers, bypassing the broken orientation heuristic
  - `_determine_weapon_arm_from_poses()` now takes `forward_sign` instead of `orientation`
  - `_run_blade_for_subject()` derives `subject_orientation` from `forward_sign`

### Blade Length Calibration Improvements (COMPLETE)
- **Body-height calibration** (primary): measures nose→ankle height, multiplied by `_BLADE_HEIGHT_RATIO=0.66`
  - Less susceptible to foreshortening than arm-based (torso/legs nearly parallel to camera)
  - Compensates for en garde stance lowering apparent height, nose-to-ankle gap vs full height
- **Segmented arm calibration** (fallback): shoulder→elbow + elbow→wrist, P90, ratio 1.8
- **WHAM 3D refinement** (Pass 2): per-frame blade scale using 3D arm foreshortening factor
  - `_get_median_3d_arm_length()`: stable 3D arm length from WHAM (skeleton doesn't change)
  - `_compute_per_frame_blade_scale()`: varies blade length per frame based on arm projection
- **TODO**: blade length still appears slightly short — revisit with fresh ideas later
  - Consider pixel-based blade detection (Hough lines in hand ROI)
  - Consider using weapon length as user input during configuration

### Opponent Blade Speeds for Action Classification (COMPLETE)
- Action classification now receives both fencer AND opponent blade speeds
- Previously opponent actions were classified without blade data
- Pipeline passes `opponent_blade_speeds` dict alongside `fencer_blade_speeds`

### Processing UI Stage Order Fix (COMPLETE)
- `ProcessingStatus.tsx` showed action_classification before blade_tracking
- Fixed to match actual pipeline order: blade_tracking → action_classification

### Two-Pass Blade Refinement Architecture (from session 13)
- Pass 1: 2D keypoints only (immediate, in main pipeline)
- Pass 2: WHAM 3D refinement (async callback after mesh reconstruction)

### Fullscreen Overlay Fixes (from session 13)
- `playerPanelRef` fullscreens entire left column (video + scrubber + timeline + controls)
- Canvas dynamically positioned for `object-contain` letterboxing

## Current State
- **Branch**: `feature/stage-3-blade-refinement`
- **All 7 containers running**: frontend(:5174), api(:8001), worker(GPU), wham(GPU), postgres(:5433), redis(:6380), ollama(:11435)
- **Blade tracking**: dual-subject, position-based forward, body-height calibration
- **Migrations applied**: through `b9c0d1e2f3a4` (blade_states subject column)
- **Migrations pending**: `c4d5e6f7a8b9` (confidence), `d5e6f7a8b9c0` (blade speed), `e6f7a8b9c0d1` (unique fencer name)

## Known Issues / Next Steps for Blade Tracking
- Blade length still slightly short in some frames — revisit later
  - Pixel-based blade detection (Hough lines, edge detection in hand region)
  - User-provided weapon length during bout configuration
  - Better WHAM 3D→2D projection for per-frame foreshortening correction
- Blade direction accuracy varies — hand keypoints still noisy at fencing video scale

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

### Frontend Updates
- 3D mesh visualization (consume WHAM mesh_states data)
- Toggle hand skeleton rendering in review UI
- Display `blade_speed_avg` / `blade_speed_peak` per action in review UI

### Future Research
- 4D Gaussian splatting for multi-camera fencing replay
- LLM coaching re-enablement (Claude API integration exists, currently stubbed)

## Commits This Session
```
2a9df00 Fix blade direction and add dual-subject blade tracking
(pending) Blade length calibration, opponent blade speeds, UI stage order fix
```

## Previous Sessions
- Session 13: WHAM 3D blade refinement, hand keypoint improvements, fullscreen overlay fix
- Session 12: WHAM end-to-end, hand keypoints, fullscreen fix, foot keypoints
- Session 11: WHAM container setup, foot keypoints started, per-fencer action timeline
- Session 10: RTMPose migration, GPU fix, configurable ports, self-contained project
- Session 9: P2c action-blade integration, P3a wrist angulation, P3b Kalman filter, strip detection, fencer housekeeping
- Session 8: P2a occlusion bridging + P2b confidence score (`296b586`)
