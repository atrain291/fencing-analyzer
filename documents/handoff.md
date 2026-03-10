# Session 13 Handoff — 2026-03-10

## What Was Done

### WHAM 3D Reconstruction — End-to-End Working (COMPLETE)
- **Commit**: `e9f4459` on `feature/stage-3-blade-refinement`
- Rewrote `wham/app/inference.py` to use actual WHAM API (`build_network`, `build_body_model`, `FeatureExtractor`)
- Staged GPU loading to fit in 12GB VRAM (4070 Super):
  1. Load HMR2a ViT backbone (~2.9 GB) → extract features frame-by-frame → release
  2. Load WHAM Network (~0.2 GB) → run inference → release
- Custom per-frame extraction loop with `torch.cuda.empty_cache()` every 50 frames
- `os.chdir(WHAM_PATH)` required before loading — WHAM uses relative config paths
- 3D joints from `network.output.joints` (internal SMPL state, not in eval output dict)
- Betas averaged across frames: `pred_betas.mean(axis=0).tolist()`
- Fixed `wham/app/tasks.py`: uses `frame_ids` to map WHAM output back to pipeline frame IDs
- Added missing Python deps to Dockerfile: `progress`, `joblib`, `chumpy`, `gdown`
- Added volume mounts in docker-compose: `wham/dataset`, `wham/checkpoints`
- SMPL model files manually downloaded and placed (non-commercial license)

### Hand Keypoint Blade Direction Detection (COMPLETE)
- **Commit**: `d5e4109` on `feature/stage-3-blade-refinement`
- Extended `pose.py` to extract 42 hand keypoints (21 per hand, `lh_`/`rh_` prefix)
  - Now stores 65 keypoints per frame (was 23: 17 body + 6 foot)
  - Hand joint names: wrist, thumb (CMC/MCP/IP/tip), index/middle/ring/pinky (MCP/PIP/DIP/tip)
- New `_compute_hand_blade_direction()` in `blade.py` with 3-method cascade:
  1. Index finger MCP→PIP direction (highest confidence)
  2. MCP line perpendicular with direction check (×0.8 confidence)
  3. Wrist→MCP midpoint (×0.6 confidence)
- Confidence-weighted blending between hand-derived and forearm-derived directions
  - Blend formula: `blend = clamp((hand_conf - 0.2) / 0.3, 0, 1)`
  - When hand confidence moderate (0.2-0.5), smoothly transitions between methods
  - When hand confidence high (>0.5), uses hand direction; skips wrist angulation correction
- Frontend `skeleton.ts`: added `HAND_SKELETON_EDGES` (23 edges per hand: finger chains + MCP line + palm center)
  - `drawSkeleton()` accepts `showHands` parameter (default false)

### Fullscreen Skeleton Overlay Fix (COMPLETE)
- **Commit**: `fa09687` on `feature/stage-3-blade-refinement`
- Bug: skeletons disappeared when video player was maximized
- Root causes:
  1. Native video fullscreen button only fullscreened `<video>`, not the container with canvas
  2. Canvas used `inset-0 w-full h-full` which misaligns with `object-contain` letterboxing
- Fixes:
  - Hidden native fullscreen button via CSS (`::-webkit-media-controls-fullscreen-button`)
  - Fullscreen now targets `playerPanelRef` (entire left column: video + scrubber + timeline + controls)
  - Flex layout in fullscreen: video grows to fill space, controls stay at bottom
  - Canvas dynamically positioned to match video's `object-contain` rendered area
  - Redirect fallback: if video itself enters fullscreen, exit and re-enter on player panel

### Pipeline Bug Fixes
- **Analysis upsert**: `duplicate key value violates unique constraint` on reprocessing — changed to upsert logic in `video_pipeline.py`
- **Broken SMPL symlinks**: container symlinks pointed to host paths — replaced with file copies
- **WHAM FP16 mismatch**: HMR2a doesn't support half precision — kept in FP32

## Current State
- **Branch**: `feature/stage-3-blade-refinement`
- **All 7 containers running**: frontend(:5174), api(:8001), worker(GPU), wham(GPU), postgres(:5433), redis(:6380), ollama(:11435)
- **Ports**: alternate ports to coexist with sister project on default ports
- **WHAM**: fully operational with staged GPU loading, SMPL models downloaded
- **Keypoints**: 65 per frame (17 body + 6 foot + 42 hand)
- **Migrations applied**: `f7a8b9c0d1e2` (action subject), `a8b9c0d1e2f3` (mesh_states)
- **Migrations pending**: `c4d5e6f7a8b9` (confidence), `d5e6f7a8b9c0` (blade speed), `e6f7a8b9c0d1` (unique fencer name)

## Branch Lineage
```
master
  └─ feature/stage-2-blade-detection (P1, P2a, P2b)
       └─ feature/stage-2c-action-blade-integration (P2c)
            └─ feature/stage-3-blade-refinement (P3a, P3b, Strip, Housekeeping, RTMPose)  <- current
```

## What's Next

### Run Pending Alembic Migrations
- 3 migrations not yet applied: confidence, blade_speed, unique fencer name

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
- Visualize detected strip polygon in configure UI
- Surface `preparation` action type in action timeline / drill report
- Display `blade_speed_avg` / `blade_speed_peak` per action in review UI

### Future Research
- 4D Gaussian splatting for multi-camera fencing replay (Arcturus, 60-90 degree camera separation)
- LLM coaching re-enablement (Claude API integration exists, currently stubbed)

## Commits This Session
```
fa09687 Fix skeleton overlay disappearing on fullscreen maximize
d5e4109 Add hand keypoint blade direction detection
e9f4459 Get WHAM 3D reconstruction working end-to-end
0aeb64c Add foot keypoints to pose estimation and skeleton rendering
```

## Previous Sessions
- Session 12: WHAM end-to-end, hand keypoints, fullscreen fix, foot keypoints
- Session 11: WHAM container setup, foot keypoints started, per-fencer action timeline
- Session 10: RTMPose migration, GPU fix, configurable ports, self-contained project
- Session 9: P2c action-blade integration, P3a wrist angulation, P3b Kalman filter, strip detection, fencer housekeeping
- Session 8: P2a occlusion bridging + P2b confidence score (`296b586`)
- Session 7: Housekeeping fixes (`21172db`, `13d1bc5`)
- Session 6: API memory leak, H.264 transcode (`60ca600`, `3248aa9`)
