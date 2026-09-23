# Pose evaluation implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline work and superpowers:subagent-driven-development for an independent delegated task. Track progress below and run regression tests before claiming completion.

**Goal:** Deliver trustworthy 2D pose comparison tooling and repair the existing identity/timing foundations.

**Architecture:** Share a PTS-preserving decoder between the worker and an independent evaluation CLI. Keep model adapters and report rendering separate. Retain the production model until real footage supports a replacement.

**Tech Stack:** Python 3.11+, PyAV, NumPy, pytest, optional Ultralytics/RF-DETR/Supervision/PyTorch; existing React/TypeScript frontend.

**Spec:** `docs/superpowers/specs/2026-09-23-pose-evaluation-design.md`

## Global constraints

- Work in `codex/pose-evaluation`; preserve the original checkout's settings edit.
- Keep production YOLOv8 weights and runtime selection unchanged by default.
- No database, cloud API, or running service required for evaluation.
- Model IDs: `yolov8n`, `yolo26m`, `rfdetr-keypoint`.
- Timestamps use media presentation time, not nominal FPS.
- Missing people/keypoints remain missing; no accuracy or identity-switch claims without labels.
- Do not commit footage, downloaded weights, or generated result files.
- Never restart/deploy the remote application while running the evaluation.

## Task 1: Worker video and participant correctness

Files: add `worker/app/pipeline/video.py`, `worker/app/pipeline/participants.py`; modify `worker/app/pipeline/pose.py`, worker dependency declarations; add `tests/test_video.py`, `tests/test_participants.py`, `tests/test_pose_pipeline.py`, `pytest.ini`.

Interfaces:

```python
@dataclass
class DecodedFrame:
    index: int
    timestamp_ms: float
    image: Any  # upright BGR ndarray

def iter_video_frames(path: str) -> Iterator[DecodedFrame]: ...

class ParticipantMap:
    def select(self, track_ids, boxes) -> tuple[int | None, int | None]: ...
```

- [x] Write tests for reordered IDs, disappeared athlete, crowd initialization, and new session reset. Hand-derived example: IDs `[10, 20]` initially map left/right; next IDs `[20]` must map `(None, 0)`.
- [x] Encode a small lossless VFR fixture with PTS `[0, 40, 120, 160]` at time base `1/1000`; assert decoded timestamps, pixels, and frame count. Test absent video and invalid timestamps.
- [x] Run focused tests and record the missing-feature failures.
- [x] Implement decoder and mapping, then wire the worker to their outputs. Preserve per-frame PTS in summaries and database records. Reset Ultralytics trackers at the beginning of each bout, and use returned box IDs to retrieve poses.
- [x] Verify frame persistence with injected model/DB boundaries so tests exercise real worker control flow without downloading weights.
- [x] Run `python -m pytest tests/test_video.py tests/test_participants.py tests/test_pose_pipeline.py -q`, inspect diff, and commit this deliverable.

## Task 2: Standalone comparison and report

Files: add `tools/pose_eval/{__init__,__main__,manifest,adapters,runner,metrics,report}.py`, `tools/pose_eval/report.html`, `evaluation/manifest.example.json`, `evaluation/README.md`, separate evaluation requirements, and matching tests; extend `.gitignore`.

Interfaces:

```python
# Normalized image coordinates; source frame dimensions retained in records.
prediction = {"box": [0.1, 0.2, 0.4, 0.9], "confidence": 0.9,
              "track_id": 10, "keypoints": {"left_wrist":
              {"x": 0.3, "y": 0.4, "confidence": 0.8}}}
record = {"frame_index": 1, "timestamp_ms": 40.0,
          "width": 640, "height": 360, "detections": [prediction]}
# Commands run from repository root:
# python -m tools.pose_eval validate evaluation/manifest.json
# python -m tools.pose_eval run evaluation/manifest.json --model yolov8n --output evaluation/results/run-1
# python -m tools.pose_eval report evaluation/results/run-1/clip-id
```

- [x] Write failing tests for manifest path resolution/duplicate IDs/invalid time ranges; independently computed visible-joint error; missing prediction denominators; report JSON escaping; and a tiny real-decoding run with injected deterministic predictions.
- [x] Implement strict manifest loading with clip ID, video path, start/end seconds, and optional annotation file. Use paths relative to the manifest, not the current directory. Reject invalid data before creating outputs or downloading models.
- [x] Implement optional adapters with lazy imports and clear capability errors. YOLO consumes BGR; RF-DETR consumes RGB and its `keypoint_confidence`, `detection_confidence`, and `data["xyxy"]` fields. Keep 17 COCO points; do not invent weapon landmarks.
- [x] Use a common Supervision ByteTrack instance for each clip/model evaluation and retain track IDs. Score identities only with manual per-frame annotation association.
- [x] Measure after warmup with CUDA synchronization where applicable. Record preprocessing+inference+postprocessing latency, total processing time separately, version/checkpoint provenance, hardware, and peak allocated CUDA memory. Persist all frame predictions and actionable errors.
- [x] Generate one report per clip from completed model runs; reject mismatched source/range/frame sequences. Embed JSON safely, share one user-selected source video, and render all panels against the same media time. Include coverage and optional labeled-joint errors, never a manufactured winner.
- [x] Run focused tests, then documented CLI smoke commands against a generated video. Keep outputs ignored and explicitly synthetic.
- [x] Document isolated GPU installation and manifest/annotation format. Commit the runnable tooling.

## Task 3: Existing overlay correctness

Files owned by this task: `frontend/src/pages/VideoReview.tsx`, new focused overlay utility/tests under `frontend/src`, and frontend test configuration/dependency declarations as needed. Do not edit Python or evaluation tooling.

Interfaces and behavior: consume existing `Frame`/`Keypoint` API types; use media timestamps in milliseconds and a calculated object-contain display rectangle. Keep `z` when interpolating. Do not infer an identity from a pose-array index.

- [x] Write focused tests for a 1920x1080 video in a 1000x1000 box (display rectangle 1000x562.5 at y=218.75), missing-pose transitions, and large timestamp gaps. Tests must execute utility/component behavior, not match source text.
- [x] Run failing tests before implementing geometry and frame sampling changes.
- [x] Draw with `requestVideoFrameCallback` using metadata.mediaTime where supported; retain a cancellable RAF fallback. Ensure only one loop, cleanup on pause/end/unmount, correct paused seeking, and repaint when data arrives.
- [x] Position and size the canvas to the visible video rectangle during normal/fullscreen playback. Clear stale skeletons when frames/video are absent. Do not bridge missing poses or gaps over 100 ms; do not extrapolate beyond the available sample window.
- [x] Fix existing build errors in the touched component (unused keypoint name and missing `z` during interpolation). Preserve slow-motion/fullscreen controls.
- [x] Run focused tests and `npm run build`; report validation and remaining limitations. Commit only task-owned files.

## Task 4: Integration and actual footage

- [x] Review completed task diffs and resolve correctness findings with focused regressions.
- [x] Run the Python suite, frontend tests/build, and `git diff --check` once on the integrated result.
- [ ] Establish SSH key access to `adeitz@omarchy`, locate authorized bout files and GPU, and run a short pilot in an isolated evaluation directory. Avoid existing app services and database writes.
- [x] If remote authentication remains unavailable, preserve exact runnable commands and report the unresolved real-footage validation explicitly. Do not label the model comparison complete.
- [x] Record results, commit verified work, and report artifact paths and the next concrete action.

## Verification record — 2026-09-23

- Python: `python -m pytest -q` — 22 passed with the pinned optional model environment installed. Includes real VFR decode, lossless rotated video with nonzero container start, invalid timestamps, worker persistence/identity mapping, adapters, ByteTrack, metrics, and report validation.
- Frontend: `npm test` — 9 passed; `npm run build` — passed. The navigation regression was demonstrated failing before the fix.
- All three official model checkpoints completed CPU smoke runs on a generated frame, both at the default threshold and at 0.000001 to exercise nonempty COCO-17 output. These are API checks only. Weights and outputs remain ignored.
- Chrome: offline report source loading, frame stepping, playback, and equal-time rendering across three synthetic panels checked; no page errors after the final report refresh. Screenshot under ignored `output/playwright/pose-comparison.png`.
- Independent review found annotation-hash comparability and stale navigation-state issues; both were fixed with regressions and re-reviewed without further findings.
- Real footage is pending SSH authentication to `adeitz@omarchy` and the source directory. No GPU benchmark, deployment/container check, or application fullscreen visual check has been completed. See `evaluation/README.md` for exact run commands and remaining limits.
