# Session 11 Handoff — 2026-03-08

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

## Current State
- **Branch**: `feature/stage-3-blade-refinement`
- **All 6 containers running**: frontend(:5173), api(:8000), worker(GPU), postgres(:5432), redis(:6379), ollama(:11434)
- **Migrations pending**: `c4d5e6f7a8b9` (confidence), `d5e6f7a8b9c0` (blade speed), `e6f7a8b9c0d1` (unique fencer name), `f7a8b9c0d1e2` (action subject)
- **RTMPose models**: auto-downloaded to `~/.cache/rtmlib/` on first inference
- **Frontend gaps**: no strip visualization, no preparation action display, no blade speed metrics

## Branch Lineage
```
master
  └─ feature/stage-2-blade-detection (P1, P2a, P2b)
       └─ feature/stage-2c-action-blade-integration (P2c)
            └─ feature/stage-3-blade-refinement (P3a, P3b, Strip, Housekeeping, RTMPose)  <- current
```

## What's Next

### Deploy & Test (HIGH PRIORITY)
- Run pending Alembic migrations
- Process a bout end-to-end with RTMPose and verify:
  - GPU utilization during inference
  - Correct skeleton overlays in review UI
  - Confidence values in 0-1 range
  - Fencer/opponent tracking stability
  - Strip detection constraining skeleton selection
  - Blade tracking, action classification still working

### Future RTMPose Opportunities (documented in rtmpose_migration.md)
- Hand keypoints (indices 91-132): improve blade tip detection from grip position
- Foot keypoints (indices 17-22): more precise footwork analysis
- Alternative tracking: custom IoU matching or standalone ByteTrack/BoT-SORT

### Frontend Updates
- Visualize detected strip polygon in configure UI
- Surface `preparation` action type in action timeline / drill report
- Display `blade_speed_avg` / `blade_speed_peak` per action in review UI

## Commits This Session
```
9574791 Add per-fencer action timeline with opponent tracking
940a064 Make project fully self-contained for multi-project coexistence
a042c00 Update handoff for session 10: RTMPose migration and GPU fixes
cbc5b62 Fix GPU inference, score normalization, configurable ports, and data paths
6fa5fc3 Replace YOLO11x-Pose with RTMPose WholeBody (rtmlib)
6f1505a Add RTMPose migration plan with tracking options and future keypoint ideas
```

## Previous Sessions
- Session 10: RTMPose migration, GPU fix, configurable ports, self-contained project, per-fencer timeline
- Session 9: P2c action-blade integration, P3a wrist angulation, P3b Kalman filter, strip detection, fencer housekeeping
- Session 8: P2a occlusion bridging + P2b confidence score (`296b586`)
- Session 7: Housekeeping fixes (`21172db`, `13d1bc5`)
- Session 6: API memory leak, H.264 transcode (`60ca600`, `3248aa9`)
