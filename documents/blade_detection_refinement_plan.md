# Blade Detection Refinement Plan

## Current State Assessment (`worker/app/pipeline/blade.py`)

### What It Does
The current blade tracking is a **pure geometric heuristic** — no actual vision-based blade detection:
1. **Weapon arm selection** (lines 37-50): Prefers right wrist (conf ≥ 0.4), falls back to left. Per-frame, no persistence.
2. **Blade direction vector** (lines 64-76): Elbow-to-wrist direction in aspect-ratio-corrected space.
3. **Blade length scaling** (lines 78-88): `shoulder-to-wrist distance × 1.5` (90cm blade / ~60cm arm). Falls back to 0.3 if shoulder confidence too low.
4. **Tip projection** (lines 90-93): Projects tip from wrist along blade direction by blade length. z=0 (no depth).
5. **Velocity** (lines 96-109): Raw frame-to-frame finite difference, no smoothing.
6. **DB persistence** (lines 111-129): Writes `BladeState` records. `nominal_xyz = tip_xyz` (no flex correction yet).

### Strengths
- Simple and fast — pure math on existing keypoints, no additional model inference
- Aspect-ratio correction prevents direction vector distortion
- Adaptive blade length based on arm proportions
- Clean pipeline integration — reads `Frame` objects, writes `BladeState` in batches

### Weaknesses
1. **No actual blade detection** — just arm extrapolation. Wrong whenever wrist angle diverges from elbow-wrist line (constantly in fencing — wrist angulation is a fundamental technique)
2. **Per-frame weapon arm selection biased to right** — left-handed fencers get wrong arm on many frames
3. **No temporal coherence** — each frame independent; wrist jitter amplified 1.5x at tip
4. **Raw finite-difference velocity** — extremely noisy at 30fps, unusable for coaching
5. **Blade angle wrong by design** — real blade controlled by wrist joint, not elbow-wrist line
6. **No occlusion handling** — any keypoint dropout creates gaps and resets velocity
7. **Variable blade length** — `arm_len × 1.5` changes with arm extension (physically wrong — blade is always 90cm)
8. **Not integrated with action classification** — `actions.py` ignores blade data entirely

### Where It Fails Most
- **During attacks/lunges**: Arm extends, wrist angulates, blade bends — all assumptions break at the most important moments
- **Side-on camera angles**: Elbow-wrist vector appears compressed/ambiguous
- **Left-handed fencers**: Wrong arm picked on many frames
- **Close-quarters actions**: Wrist/elbow confidence drops from occlusion, creating gaps when fencers are most engaged

---

## Refinement Priorities

### Priority 1 — Do Now (High Value, Low Effort, ~4 hours total)

#### 1a. Persistent Weapon Arm Detection
- **What**: Determine weapon arm once from fencer orientation (reuse `actions.py:_detect_orientation()`) or fencer `preferences` JSON, instead of per-frame right-wrist bias
- **Impact**: Eliminates wrong-arm picks for left-handed fencers and reduces frame-to-frame arm switching noise
- **Files**: `blade.py` (accept weapon_arm param), `video_pipeline.py` (pass it), optionally `fencer.py` (add handedness to preferences)
- **Effort**: ~1-2 hours

#### 1b. Temporal Smoothing (EMA)
- **What**: Apply exponential moving average to tip position before velocity calculation: `smoothed = α × raw + (1-α) × prev_smoothed` with α ≈ 0.3-0.5
- **Impact**: Makes tip trail and speed readout immediately usable; current raw values too noisy for coaching
- **Files**: `blade.py` (~10-15 lines)
- **Effort**: ~1 hour

#### 1c. Fixed Blade Length
- **What**: Measure arm length at full extension in early frames, use fixed ratio for entire bout. Stop blade from appearing to grow/shrink with arm extension.
- **Impact**: Physically correct blade projection — real epee is always 90cm
- **Files**: `blade.py` (add calibration pass at start of `run_blade_tracking()`)
- **Effort**: ~2 hours

### Priority 2 — Do Soon (Medium Value, Medium Effort)

#### 2a. Occlusion Bridging via Linear Interpolation
- **What**: Instead of skipping frames where wrist/elbow confidence drops, buffer last N valid positions. If gap < ~5 frames (150ms at 30fps), linearly interpolate. If gap exceeds threshold, emit nothing.
- **Impact**: Eliminates gaps in blade trail during fast movements; preserves data during attacks
- **Files**: `blade.py` (gap-bridging state machine)
- **Effort**: ~3-4 hours

#### 2b. Blade Visibility Confidence Score
- **What**: Composite confidence from: wrist/elbow/shoulder keypoint confidence, temporal consistency (tip position jump from previous frame), arm extension ratio
- **Impact**: Frontend can modulate blade overlay opacity; coaching can qualify blade position statements
- **Files**: `blade.py`, optionally add `confidence` to `BladeState` model, `VideoReview.tsx` (opacity modulation in `drawBlade()`)
- **Effort**: ~3-4 hours

#### 2c. Action Classification Integration
- **What**: Feed blade speed into lunge detection as supplementary signal; compute per-action blade metrics (path linearity, tip speed profile); distinguish attacks from preparations
- **Impact**: Better action classification; richer drill report with blade metrics
- **Files**: `actions.py` (accept blade data), `video_pipeline.py` (pass blade data to classification), optionally extend Action model
- **Effort**: ~5-8 hours
- **Dependencies**: Requires 1b (EMA smoothing) so blade speed values are reliable

### Priority 3 — Significant Upgrade (High Value, Significant Effort)

#### 3a. Wrist Angulation Correction
- **What**: Blend between elbow→wrist and shoulder→wrist vectors based on arm extension ratio. Extended arm → blade aligns more with arm line; bent arm → infer blade angle from fencing-specific priors (en garde wrist angle inward)
- **Impact**: Largest single accuracy improvement within pure keypoint math; addresses the fundamental weakness of equating forearm axis with blade direction
- **Files**: `blade.py` (new wrist angle estimation function, modified tip projection)
- **Effort**: ~8-12 hours including testing/tuning
- **Dependencies**: Persistent weapon arm detection (1a)

#### 3b. Kalman Filter for Tip State Estimation
- **What**: Replace EMA + occlusion bridging with proper state estimator. State vector: `[tip_x, tip_y, vel_x, vel_y]`. Measurement: raw projected tip. Process model: constant velocity. Measurement noise scaled by inverse keypoint confidence. Predicts through occlusions.
- **Impact**: Subsumes refinements 1b (EMA) and 2a (occlusion bridging); optimal velocity estimates
- **Files**: `blade.py` (replace frame loop with Kalman filter), add `filterpy` or `scipy.signal` dependency
- **Effort**: ~6-10 hours (well-understood algorithm but requires noise covariance tuning)
- **Note**: Do INSTEAD of 1b + 2a, not in addition to them

### Priority 4 — Defer to Stage 3+ (High Effort, Requires Infrastructure)

#### 4a. Guard/Bell Guard Detection (Custom YOLO Object Detector)
- **What**: Train custom YOLOv8 to detect the bell guard. Guard orientation provides actual blade axis independent of elbow position. Blade axis becomes `guard center → wrist → tip`.
- **Impact**: Fundamentally better blade axis; this is what the architecture doc planned for Stage 2 (Section 3.1) but was skipped
- **Files**: New `guard_detection.py`, modified `blade.py`, `video_pipeline.py` (new stage)
- **Effort**: Large — requires labeled training data (see architecture doc Section 11: "Blade training data scarcity")
- **Note**: Consider Roboflow for initial labeling from user's own footage

#### 4b. Color/Edge-Based Visual Detection
- **What**: OpenCV Hough line detection + color segmentation for metallic blade in wrist ROI. Validates or corrects keypoint-based estimate.
- **Impact**: Actual visual confirmation of blade position
- **Files**: `blade.py`, requires pipeline restructuring to pass pixel data (currently receives only `Frame` ORM objects)
- **Effort**: Large — pipeline restructure + high false positive risk (masks, metallic lamé, reflections)
- **Dependencies**: Priority 1-3 geometric refinements first; geometric estimate serves as search prior for visual detector

#### 4c. ML-Based Blade Tip Detection
- **What**: Fine-tune keypoint detector with "blade_tip" as 18th keypoint, or train specialized tip regression model on weapon arm crop
- **Impact**: Most accurate approach
- **Effort**: Very large — blade tip is ~1 pixel in 1080p; substantial training data requirements
- **Dependencies**: Guard detection (4a) more practical first since guard is larger/more reliably detectable

#### 4d. Blade Flex Physics (Architecture Doc Section 4.2)
- **What**: Euler-Bernoulli cantilever beam model with 3-5 nodes along blade. Flex proportional to lateral acceleration of guard/wrist. `flex_offset_xyz` already exists in BladeState schema.
- **Impact**: Accurate tip position during impacts and fast movements
- **Dependencies**: Guard detection (4a) for accurate blade axis + depth estimation for 3D

---

## Architecture Doc Gap Analysis

### Planned vs. Implemented (Stage 2: Weapon Tracking)

The architecture doc envisioned this tracking hierarchy:
1. ✅ Wrist keypoint (high confidence) — implemented
2. ❌ Guard/shield detected by custom YOLOv8 — **not implemented**
3. ❌ Blade axis from wrist-to-guard geometry — **cannot do without guard**
4. ⚠️ Nominal tip projected at fixed 90cm — implemented but with VARIABLE length
5. ❌ Blade flex correction via physics — Stage 3 (schema fields `flex_offset_xyz`, `nominal_xyz` ready)
6. ❌ Final 3D tip = nominal + flex offset — Stage 3 (needs depth estimation)

### Stage 3 Dependencies
- **Depth Anything V2** for monocular depth estimation
- **3D promotion** of all keypoints
- **Guard detection** (ideally from Stage 2)
- **Blade flex physics** requires accurate blade axis (guard detection) + 3D positions

### Stage 4 Impact
Accurate blade tracking is foundational for Stage 4 threat analysis:
- Correction cost model (angular calculation from blade tip to target)
- Closing velocity
- Cone of commitment visualization
- Point-of-no-return detection
- Effective Distance Penalty (Section 4.5) — attack inefficiency expressed as cm of strip disadvantage

---

## Critical Files for Implementation

| File | Role |
|------|------|
| `worker/app/pipeline/blade.py` | Core blade tracking — every Priority 1-3 change touches this |
| `worker/app/pipeline/actions.py` | Action classification; has `_detect_orientation()` reusable for weapon arm; needs blade data integration |
| `worker/app/tasks/video_pipeline.py` | Pipeline orchestrator; needs modification for passing blade data to action classification |
| `frontend/src/pages/VideoReview.tsx` | Blade visualization (`drawBlade()` at line 60-85); needs confidence-based opacity |
| `worker/app/models/analysis.py` | BladeState schema — already has `flex_offset_xyz`, `nominal_xyz`, `correction_cost`; may need `confidence` column |
| `backend/app/models/analysis.py` | Mirror of worker BladeState model |

---

*Document created: 2026-03-03, based on analysis of current codebase on branch `feature/stage-2-blade-detection`*
