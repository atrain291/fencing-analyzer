# Omarchy evaluation runtime — 2026-09-23

SSH key access works for `adeitz@omarchy`. The isolated checkout is `/home/adeitz/source/fencing-pose-evaluation`, on `codex/pose-evaluation`. Model code was validated at commit `06b0f95590f05a81127f2c7afbe422a7ebbd5f6d`.

| Component | Validated configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB VRAM |
| Driver | 610.57.04 |
| System | Linux 7.2.3-arch1-3, glibc 2.44 |
| Evaluation Python | 3.12.14, managed by uv; `.venv-eval` |
| PyTorch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 |
| CUDA runtime | 12.8; inference device `cuda:0` |
| Model packages | Ultralytics 8.4.160, RF-DETR 1.10.1 |
| Decoder / tracker | PyAV 18.1.0, Supervision 0.30.5 |

The system Python 3.14 installation was left unchanged. The matched GPU wheels follow the [official PyTorch 2.8 installation combinations](https://pytorch.org/get-started/previous-versions/#v280). NVIDIA's package endpoint timed out during installation, so dependencies came from PyPI and the two PyTorch wheels came directly from their official wheel index with its published SHA-256 hashes.

## Checks performed

- All 22 Python regression tests passed on this Linux environment.
- YOLOv8n-Pose, YOLO26m-Pose, and RF-DETR Keypoint Preview each completed a CUDA run on the generated source video: three warmup predictions and one measured frame, at the default 0.25 detection threshold.
- Separate runs at a deliberately tiny threshold of 0.000001 exercised nonempty predictions and COCO-17 conversion for every model. CUDA allocation was positive and recorded in all six results.
- Source-video and checkpoint SHA-256 values matched the earlier Windows CPU checks.

These are runtime and API checks, **not fencing accuracy or throughput benchmarks**. One generated frame cannot rank the models. The tiny threshold is only an integration test setting.

The six result JSON files are preserved on Omarchy under `evaluation/results/gpu-smoke/smoke/` and `evaluation/results/gpu-api-smoke/smoke/`. Local copies are under `evaluation/results/omarchy-validation/`. These generated files and the model weights remain excluded from Git.

## Resume with real footage

The source folder is still needed. The historical `/run/media/adeitz/63A504213DF71637/fencing-visualizer` path no longer exists. Once the user supplies the folder, select a clear exchange and a difficult exchange, confirm their timestamps visually, and follow the [pilot instructions](README.md).

```sh
ssh adeitz@omarchy
cd /home/adeitz/source/fencing-pose-evaluation
source .venv-eval/bin/activate
python -m tools.pose_eval validate evaluation/manifest.local.json
```

The local manifest must first be created with the chosen footage. Reuse the official checkpoints already present in `evaluation/results/smoke-weights/` through `--weights`. Keep the 0.25 detection threshold for the initial pilot. No production services or databases were changed, and the application fixes have not been deployed.
