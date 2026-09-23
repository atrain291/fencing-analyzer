# First pose evaluation milestone

The user authorized the next steps in `documents/model-research-and-evaluation-plan-2026-09-22.md`. Build a reproducible 2D comparison before choosing a production model. Videos are on Omarchy; SSH access is being established. Continue implementation and local synthetic verification independently of that access.

## Approach

Keep the production model unchanged. Correct its timestamp and participant association foundations, then add a standalone evaluation command that does not require the database, Redis, Celery, or an LLM. This supports separate evaluation environments and avoids migrating the deployed ML stack before a winner is measured. A production-model replacement now would hide whether gains came from correctness fixes or the network. A notebook alone would make repeatable exports and testing harder.

The evaluation supports YOLOv8 nano, YOLO26 medium, and RF-DETR Keypoint Preview through small adapters. It decodes each clip using presentation timestamps, records predictions with provenance and timing, and generates an offline comparison report with a shared source-video clock. Ground-truth annotations are optional: accuracy and identity-switch results are unavailable without labels, never inferred from confidence scores.

## Data and boundaries

- Video decoding yields `DecodedFrame(index, timestamp_ms, image)` where image is upright BGR and timestamps are relative to the media timeline. Preserve variable intervals and reject missing/non-monotonic timestamps rather than silently inventing a nominal frame rate. Rotation is applied before inference.
- A detection carries a box, confidence, COCO keypoints, and optional tracker ID. Keep missing observations missing.
- Worker participant mapping uses persistent tracker IDs. Initialize only from exactly two tracked people, ordered left/right; do not pick two out of a crowd or reassign a missing athlete. This conservative convention is not verification of the uploaded profile's identity. Explicit selection remains necessary for general crowded footage.
- Evaluation saves all people and uses stable per-model track IDs, with explicit identity annotation for scoring. It does not call detection-array positions participant identities.
- Model outputs and local videos stay in ignored result directories. Reports state whether they contain real inference or synthetic fixtures.
- A report uses one video element and the same displayed timestamp for every model panel. Aspect-ratio fit and missing poses are preserved. It must not require a hosted service.

## Scope and acceptance

1. Regression tests demonstrate correct worker identity after detection reordering/disappearance and no association leakage between bouts.
2. Real encoded synthetic variable-frame-rate video verifies timestamps and decoder cleanup/errors without ML inference.
3. CLI validates manifests and model capabilities before inference; has separate model runs and report aggregation; records package versions, checkpoint identifiers, hardware, resolution, model latency, complete run time, and peak CUDA allocation where available.
4. Report supports synchronized seeking, visible detection/track labels, and source-file selection. It exposes coverage and labeled-joint error when annotations exist and clearly marks unavailable metrics.
5. Existing review overlay fits letterboxed video and follows decoded video frames; no interpolation across missing poses or large observation gaps. Preserve fullscreen and slow-motion controls.
6. Local unit/integration checks and frontend build pass. Run a real clip on Omarchy only once access and representative footage are available. Do not claim real model validation from fixtures.

3D reconstruction and coaching-event validation are later experiments after this 2D foundation. Do not deploy or restart existing Omarchy services as part of the comparison.
