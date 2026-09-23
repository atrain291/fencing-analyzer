# Evaluate the pose models on existing bout videos

This pilot runs independently of the application, database, queue, and LLM. Keep the production worker on its current YOLOv8 model until the measurements justify changing it. Start with short, single-shot clips where both athletes are visible. Code is ready for evaluation; real fencing accuracy has not been established by the synthetic regression fixtures.

## Install in a separate environment

Use Python **3.11 or newer**. On the GPU host, create a fresh virtual environment and install a matching PyTorch/torchvision CUDA build for its driver using the [official selector](https://pytorch.org/get-started/locally/). Do not upgrade packages inside the running worker container.

```sh
python3 -m venv .venv-eval
. .venv-eval/bin/activate
# Install the matched torch/torchvision GPU packages first.
python -m pip install -r evaluation/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

On Windows, use `.venv-eval\Scripts\Activate.ps1`. The CLI accepts `--device cpu` for a functional smoke run; CPU times are not an estimate of GPU throughput. First model construction downloads the official pretrained weights. Alternatively pass a locally available checkpoint with `--weights /path/to/checkpoint`. The checkpoint hash is included when its resolved file is available.

The pinned evaluation packages contain YOLO26 and RF-DETR's preview keypoint API. RF-DETR is a preview, so future package upgrades should be validated against the adapter tests. The production Dockerfile retains Ultralytics 8.3.0; both production and evaluation decoders now use PyAV 18.1.0 (Python 3.11+).

## Select the footage and run

Copy `evaluation/manifest.example.json` to `evaluation/manifest.local.json`, replace the example paths, and select short start/end intervals. Paths are local to the execution machine and resolve relative to the manifest. End times are exclusive. Each clip needs a unique ID containing letters, digits, hyphens, or underscores. A video cut starts a new clip; track IDs are reset between clips.

From the repository root:

```sh
python -m tools.pose_eval validate evaluation/manifest.local.json
python -m tools.pose_eval run evaluation/manifest.local.json --model yolov8n --output evaluation/results/pilot-1
python -m tools.pose_eval run evaluation/manifest.local.json --model yolo26m --output evaluation/results/pilot-1
python -m tools.pose_eval run evaluation/manifest.local.json --model rfdetr-keypoint --output evaluation/results/pilot-1
python -m tools.pose_eval report evaluation/results/pilot-1/clear-footwork
```

Run sequentially on the same idle GPU. Each command uses a separate process so previous models do not occupy GPU memory. Defaults are 640 pixels for YOLO and 576 for RF-DETR; resolutions are recorded and can be set with `--image-size`. This evaluates practical native configurations, not a claim of identical pixel budgets. Inference uses float32. Detection threshold defaults to 0.25. Three warmup predictions precede measurement and do not consume evaluation frames.

The YOLOv8 weights are the original architecture baseline run in the same current evaluation library as YOLO26. This is not a reproduction of the entire old Ultralytics 8.3.0 worker's speed. Keep that distinction in reports.

An existing model result is never overwritten. Choose a new output directory for a rerun. Completed earlier clips remain available if a later clip fails; fix the input and select the remaining clips in a new manifest. Results and local manifests are git-ignored. Model initialization, annotation validation, or decoding errors fail the command rather than producing a successful empty run.

Open each clip's `comparison.html` and select the **same original video**. No upload or hosted service is involved. One video drives all panels, with previous/next evaluated-frame controls and slow playback. The browser checks file size and upright dimensions; it does not rehash the full video. The source SHA-256 is available in the provenance section. Use a browser-compatible MP4; if conversion is necessary, evaluate the converted file itself so that the report and predictions refer to the same timestamps.

## What is measured

- Prediction latency includes preprocessing, model inference, and conversion to the common output schema. CUDA work is synchronized. Tracking time is measured separately. These are Python/PyTorch measurements, not the published TensorRT latency numbers.
- Processing time includes sequential decoding from the beginning of the file, prediction, and tracking. It excludes model setup/download, warmup, source hashing, metrics, and exports. Clip offsets can therefore affect processing time; compare the same clip across models. Setup and warmup are recorded separately.
- Peak CUDA allocation is measured with the model loaded after warmup. It is PyTorch allocated memory, not all system GPU memory. CPU runs report it as unavailable.
- Coverage is the fraction of frames with at least two detected people. Referees/background fencers can increase it; it is not accuracy.
- Accuracy requires manually labeled visible joints and explicit per-model identity associations. Missing annotations never become a perfect score. Jitter, action-boundary accuracy, and true 3D accuracy are not measured by this pilot.

Every run saves source identity/range, original frame indices and PTS, dimensions, all detections, track IDs, keypoint confidence, RF-DETR covariance when supplied, model/checkpoint and package versions, device, and timings. ByteTrack uses the same settings for each model. Its IDs are per-run tracklets, not verified athlete identities. Fragmentation and swaps still need evaluation. The tracker uses a fixed frame-rate setting of 30 for its frame-based lifetime; this does not resample the video.

## Optional manual accuracy labels

Add `"annotations": "labels.local.json"` to a clip. Coordinates are normalized to the upright original image. Copy original frame indices and each model's displayed track ID from the report. A `null` track ID means a confirmed missed person; an omitted model entry means that model has not been annotated. Do not estimate hidden joint coordinates.

```json
[
  {
    "frame_index": 600,
    "people": [
      {
        "person_id": "athlete-a",
        "track_ids": {"yolov8n": 1, "yolo26m": 3, "rfdetr-keypoint": null},
        "keypoints": {
          "left_wrist": {"x": 0.31, "y": 0.48, "visible": true},
          "right_wrist": {"visible": false}
        }
      }
    ]
  }
]
```

Run into a new result directory with these labels. Use the same annotation file for every model: the report rejects different annotation hashes, including labeled versus unlabeled runs. Mean joint error is measured in source pixels only for matched, visible joints above confidence 0.30; visible-joint recall includes confirmed missing predictions in its denominator. Interpret the two together. Labeled ID changes count different track IDs assigned to the same manually named person between labeled observations. This is a pilot metric, not a standardized MOT score. Check the displayed matched/labeled joint counts for comparable annotation coverage before comparing models.

## Verification and current limitations

```sh
python -m pip install -r evaluation/requirements-test.txt
python -m pytest -q
cd frontend
npm ci
npm test
npm run build
```

The Python regression suite generates real lossless VFR and rotated, nonzero-start fixtures to test decoder and export behavior, then injects deterministic predictions at the external-model boundary. Those synthetic predictions are explicitly labeled and do not test neural-network quality. Adapter tests exercise schema conversion without loading weights; the real ByteTrack test runs when the optional evaluation dependencies are installed.

On 2026-09-23, all three official checkpoints completed one-frame CPU smoke runs on Windows with Python 3.12.14 and PyTorch 2.8.0+cpu/torchvision 0.23.0+cpu. Additional runs at an intentionally tiny detection threshold exercised nonempty output conversion. These generated-image runs validate API integration only, not detection quality or GPU speed. The offline report was also checked in Chrome for synchronized playback and frame stepping. A GPU run on representative fencing footage, production-container validation, and a visual fullscreen check of the existing application remain outstanding.

Supervision 0.30.5 warns that its ByteTrack implementation will be removed in 0.31.0, and Ultralytics warns that `half` will be replaced by `quantize`. Keep the tested pins for this pilot; revalidate those APIs when upgrading.

The worker now decodes in software to preserve timestamps, replacing the timestamp-discarding raw NVDEC pipe. Measure this end-to-end trade-off before deployment. Its conservative participant map initializes only when exactly two tracked people are present and labels them left/right at that moment; it does not identify the uploaded fencer profile, choose among a crowd, or recover lost IDs automatically. Full participant selection/correction remains future work. No live services are restarted by these tools.
