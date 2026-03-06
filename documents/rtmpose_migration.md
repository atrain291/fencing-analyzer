# RTMPose WholeBody Migration Plan

## Decision: Replace YOLO11x-Pose with rtmlib RTMPose WholeBody

### Why
- RTMPose WholeBody provides 133 keypoints (vs YOLO's 17)
- First 17 keypoints are identical to COCO-17 format (same indices, same joint names)
- Hand keypoints (21 per hand) can improve blade tracking
- Foot keypoints (3 per foot) can improve footwork analysis

### Tracking Approach

**Chosen: Option 1 — rtmlib PoseTracker (IoU-based)**
- Use rtmlib's built-in PoseTracker for frame-to-frame person tracking
- Keep existing custom ID locking/re-lock logic on top of tracker IDs
- Simplest path, existing occlusion recovery logic already handles gaps

**Deferred: Option 2 — Custom IoU matching**
- Use raw Wholebody detections (no PoseTracker)
- Implement our own bbox IoU matching across frames
- More control over matching thresholds and logic
- Consider if PoseTracker proves too unreliable for fencing (fast lateral movement, lunges)

**Deferred: Option 3 — Dedicated standalone tracker (ByteTrack/BoT-SORT)**
- Add a dedicated multi-object tracker library alongside rtmlib
- Best tracking quality (Kalman prediction, re-identification)
- More complex integration, additional dependency
- Consider if occlusion recovery becomes a persistent problem

### Future Opportunities (from 133 keypoints)

**Hand keypoints (indices 91-132):**
- 21 keypoints per hand (fingertips, knuckles, palm)
- Could replace geometric wrist-based blade tip projection with actual grip detection
- Finger extension/flexion could indicate grip pressure or weapon manipulation
- Priority: evaluate after initial migration is stable

**Foot keypoints (indices 17-22):**
- Big toe, small toe, heel per foot (6 total)
- More precise footwork detection than ankle-only
- Could detect toe-first vs heel-first landing (advance vs retreat biomechanics)
- Heel position useful for lunge depth measurement

**Face keypoints (indices 23-90):**
- 68 facial landmarks
- Low priority for fencing analysis
- Could potentially detect mask on/off state

### rtmlib API Reference

```python
from rtmlib import Wholebody, PoseTracker, draw_skeleton

# Single frame detection
wholebody = Wholebody(mode='performance', backend='onnxruntime', device='cuda')
keypoints, scores = wholebody(frame)  # (N, 133, 2), (N, 133)

# Video with tracking
pose_tracker = PoseTracker(
    Wholebody,
    mode='performance',
    det_frequency=10,
    tracking=True,
    tracking_thr=0.3,
    backend='onnxruntime',
    device='cuda'
)
keypoints, scores = pose_tracker(frame)
```

### Model Variants
| Mode | Detector | Pose Model | Notes |
|------|----------|-----------|-------|
| performance | YOLOX-m 640x640 | rtmw-dw-x-l 384x288 | Best accuracy |
| balanced | YOLOX-m 640x640 | rtmw-dw-x-l 256x192 | Good tradeoff |
| lightweight | YOLOX-tiny 416x416 | rtmw-dw-l-m 256x192 | Fastest |
