# Session 12 Handoff — 2026-03-09

## What Was Done

### RTMPose WholeBody Migration (COMPLETE)
- **Commits**: `6fa5fc3`, `cbc5b62` on `feature/stage-3-blade-refinement`
- Replaced YOLO11x-Pose (ultralytics) with RTMPose WholeBody (rtmlib)
- RTMPose provides 133 keypoints (17 body + 6 feet + 68 face + 42 hands)
- First 17 body keypoints are identical to COCO-17 — all downstream code unchanged
- Proximity-based person matching replaces BoT-SORT tracker IDs:
  - Initial lock: user ROI bbox from preview
  - Frame-to-frame: closest center distance (max 15% of frame diagonal)
  - Re-lock after occlusion: expanded search with 3-frame confirmation gate
- `_normalize_score()` sigmoid converts SimCC logits (0-8 range) to 0-1 probabilities
- Removed: `botsort_fencing.yaml`, `export_tensorrt.py`
- Migration plan with deferred options: `documents/rtmpose_migration.md`

### GPU Inference Fix (COMPLETE)
- rtmlib depends on `onnxruntime` (CPU) which shadows `onnxruntime-gpu`
- Fix: uninstall CPU version before installing GPU version in Dockerfile
- Added `LD_LIBRARY_PATH` for nvidia pip package CUDA/cuDNN shared libs
- Verified: `CUDAExecutionProvider` + `TensorrtExecutionProvider` available

### Configurable Ports (COMPLETE)
- Added `.env` variables: `FRONTEND_PORT`, `API_PORT`, `POSTGRES_PORT`, `REDIS_PORT`, `OLLAMA_PORT`
- `docker-compose.yml` uses `${VAR:-default}` substitution for host port mappings
- Container-internal ports unchanged; only host-side mappings are configurable

### Data Path Fix (COMPLETE)
- Updated postgres/redis bind mounts from old external drive UUID to `/mnt/data/fencing-data/`

### Per-Fencer Action Timeline (COMPLETE)
- **Commit**: `9574791` on `feature/stage-3-blade-refinement`
- Added `subject` column ("fencer" / "opponent") to Action model + Alembic migration `f7a8b9c0d1e2`
- Refactored `actions.py`: extracted `_classify_subject()` helper, runs on both `fencer_pose` and `opponent_pose`
- Opponent forward direction inverted (they face opposite to fencer); no blade data for opponent
- Frontend dual-track timeline: "You" + "Opp" rows with shared legend, tooltip shows subject
- Drill report scoped to fencer actions only
- Side panel `activeAction` filtered to fencer only

### WHAM 3D Mesh Reconstruction (COMPLETE — container built, checkpoints pending)
- **Commit**: `1c7b6c4` on `feature/stage-3-blade-refinement`
- Separate container (`fencing-analyzer-wham`) with PyTorch 1.13 + CUDA 11.6 (incompatible with main worker's PyTorch 2.4.1)
- Fire-and-forget async dispatch from main pipeline after pose estimation
- Runs per-subject (fencer + opponent separately) — WHAM is single-person model
- Uses `--estimate_local_only` (no DPVO) — global trajectory from piste homography instead
- Outputs: SMPL body pose, shape (betas), 3D joint positions (24 SMPL joints), foot contact probabilities
- New `mesh_states` table + Alembic migration `a8b9c0d1e2f3`
- Frontend types updated (MeshState interface, mesh_states on Frame)
- **Pending**: SMPL model files (manual registration at smpl.is.tue.mpg.de) + WHAM checkpoints (gdown rate-limited)

### Foot Keypoints (COMPLETE)
- Extended `KEYPOINT_NAMES` in `pose.py` from 17 to 23 (added RTMPose indices 17-22: big_toe, small_toe, heel for each foot)
- Added 6 foot bone edges to `SKELETON_EDGES` in `skeleton.ts` (ankle→heel, ankle→big_toe, big_toe→small_toe)
- Zero-cost improvement — RTMPose was already detecting these, we were just discarding them

## Current State
- **Branch**: `feature/stage-3-blade-refinement`
- **All 7 containers running**: frontend(:5174), api(:8001), worker(GPU), wham(GPU), postgres(:5433), redis(:6380), ollama(:11435)
- **Ports**: alternate ports to coexist with sister project on default ports
- **Migrations applied**: `f7a8b9c0d1e2` (action subject), `a8b9c0d1e2f3` (mesh_states)
- **Migrations pending**: `c4d5e6f7a8b9` (confidence), `d5e6f7a8b9c0` (blade speed), `e6f7a8b9c0d1` (unique fencer name)
- **RTMPose models**: auto-downloaded to `~/.cache/rtmlib/` on first inference
- **Frontend gaps**: no strip visualization, no preparation action display, no blade speed metrics, no 3D mesh visualization

## Branch Lineage
```
master
  └─ feature/stage-2-blade-detection (P1, P2a, P2b)
       └─ feature/stage-2c-action-blade-integration (P2c)
            └─ feature/stage-3-blade-refinement (P3a, P3b, Strip, Housekeeping, RTMPose)  <- current
```

## What's Next

### WHAM Setup (HIGH PRIORITY)
- Register at smpl.is.tue.mpg.de and download SMPL body models
- Place in `wham/dataset/body_models/` per WHAM README
- Manually download WHAM checkpoints (gdown rate-limited during build)
- Test WHAM inference end-to-end

### Deploy & Test
- Process a bout end-to-end and verify:
  - Foot keypoints render correctly in review UI
  - Dual-track action timeline shows fencer + opponent
  - GPU utilization during inference
  - Fencer/opponent tracking stability

### Future RTMPose Opportunities (documented in rtmpose_migration.md)
- Hand keypoints (indices 91-132): improve blade tip detection from grip position
- Alternative tracking: custom IoU matching or standalone ByteTrack/BoT-SORT

### Frontend Updates
- 3D mesh visualization (consume WHAM mesh_states data)
- Visualize detected strip polygon in configure UI
- Surface `preparation` action type in action timeline / drill report
- Display `blade_speed_avg` / `blade_speed_peak` per action in review UI

### Future Research
- 4D Gaussian splatting for multi-camera fencing replay (Arcturus, 60-90 degree camera separation)
- `pipeline/homography.py` for strip-based world coordinate correction
- Populate `kinetic_states` and `threat_metrics` from WHAM 3D data

## Commits This Session
```
1c7b6c4 Add WHAM 3D mesh reconstruction service
8b12856 Update handoff for session 11
9574791 Add per-fencer action timeline with opponent tracking
(+ foot keypoints commit pending)
```

## Previous Sessions
- Session 11: WHAM 3D mesh reconstruction, foot keypoints, per-fencer action timeline
- Session 10: RTMPose migration, GPU fix, configurable ports, self-contained project
- Session 9: P2c action-blade integration, P3a wrist angulation, P3b Kalman filter, strip detection, fencer housekeeping
- Session 8: P2a occlusion bridging + P2b confidence score (`296b586`)
- Session 7: Housekeeping fixes (`21172db`, `13d1bc5`)
- Session 6: API memory leak, H.264 transcode (`60ca600`, `3248aa9`)
