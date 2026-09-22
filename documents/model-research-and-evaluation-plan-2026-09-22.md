# Model research and proposed evaluation — 22 September 2026

This document preserves the follow-up research and recommended next steps from the September 2026 discussion. The target is **existing single-camera bout videos**, with an emphasis on épée coaching. Read the [code and technology review](technology-review-2026-09-22.md) for the current implementation, known correctness issues, and the wider architecture assessment.

Status: research and a proposed evaluation, not an implemented upgrade. No candidate was run on this project's footage. Published results below establish promising options; they do not establish fencing accuracy or performance on the intended deployment GPU. Release status and documentation were checked during the September 22 research session and should be checked again when implementation begins.

Code baseline: the accompanying review examined `9ca863f`. Newer remote commits through `4898c71` were integrated when publishing this research on September 23. Recheck the current implementation before acting on the proposed correctness fixes; this document does not constitute a review of those incoming changes.

## Main finding

There is evidence of both improvements the user asked about: stronger predictions and more useful outputs, and comparable capability with less computation. The updated first comparison should include the existing YOLOv8 nano, YOLO26 medium, and RF-DETR Keypoint Preview. Human3R and Fast SAM 3D Body are the leading follow-up 3D experiments. This ordering updates the initial shortlist in the code review; RTMPose/RTMW and ViTPose++ remain alternatives if these candidates leave important errors.

## A comparable pose benchmark

Roboflow re-evaluates the following models on COCO val2017 using the same protocol. Latency uses NVIDIA T4, TensorRT 10.4, CUDA 12.4, FP16, batch size one. These are vendor-run measurements of individual inference latency, with a 200 ms buffer between passes, rather than sustained video throughput. AP is a benchmark score, not the percentage of joints that will be correct in fencing.

| Model | Keypoint AP50:95, higher is better | Latency | Parameters |
|---|---:|---:|---:|
| YOLO11 extra-large | 68.6 | 10.6 ms | 58.8 million |
| YOLO26 medium | 68.0 | 4.6 ms | 21.5 million |
| YOLO26 extra-large | 71.0 | 9.8 ms | 57.6 million |
| RF-DETR Keypoint Preview | 71.8 | 9.7 ms | 40.7 million |

YOLO26 medium nearly matches YOLO11 extra-large's score with approximately 57% lower latency and 63% fewer parameters. YOLO11 is a generational comparison here; the project's actual baseline is YOLOv8 nano. Do not combine this table with latency figures measured on other hardware. [Benchmark source and methodology](https://rfdetr.roboflow.com/latest/learn/benchmarks/)

## Candidates and their proposed roles

### YOLO26 pose: practical first upgrade

YOLO26 is a released 2026 family available through Ultralytics. The familiar interface makes it a relatively contained comparison with the current worker. Evaluate medium for accuracy and optionally nano or small if resource use becomes the constraint. The repository's pinned Ultralytics runtime must be upgraded deliberately; changing a checkpoint filename is not a complete migration. [Model documentation](https://docs.ultralytics.com/models/yolo26/), [pose usage](https://docs.ultralytics.com/tasks/pose/)

### RF-DETR Keypoint Preview: uncertainty and custom landmarks

Released in June 2026, this model predicts positional covariance for each keypoint, providing an uncertainty ellipse as well as visibility-related signals. That gives downstream tracking and coaching a way to weight ambiguous wrist or elbow estimates. Its uncertainty calibration still needs testing on fencing.

The pretrained model predicts 17 COCO body landmarks. Custom keypoint training could be investigated for the guard and visible blade tip; those are not supplied by the pretrained checkpoint. Code and weights are available through the `rfdetr` Python package under Apache 2.0. It remains an early-access preview whose API and weights may change. The current usage documentation says it is not yet available through the separate `inference` package. [Technical explanation](https://blog.roboflow.com/real-time-keypoint-detection-with-rf-detr/), [usage and release limitations](https://rfdetr.roboflow.com/latest/learn/run/keypoints/)

### Fast SAM 3D Body: less expensive body reconstruction

This March 2026 work accelerates SAM 3D Body by parallelizing crops, pruning redundant operations, and compiling GPU execution. Its automatic pipeline reports roughly 8–11 times higher throughput on an RTX 6000 Ada. Accuracy is broadly similar but not identical: some reported measures regress. A separate deployment reports approximately 65 ms per frame on an RTX 5090. These are different hardware/settings, not interchangeable figures.

The much larger advertised speedup for MHR-to-SMPL conversion refers to one conversion stage, not the entire reconstruction pipeline. Code and checkpoint setup instructions are released. This is a candidate for detailed body reconstruction on selected exchanges; it does not provide weapon reconstruction or automatically guarantee coherent motion between frames. [Paper and benchmark conditions](https://arxiv.org/html/2603.15603v1), [implementation](https://github.com/yangtiming/Fast-SAM-3D-Body)

### Human3R: people, scene, and camera in one model

Human3R, introduced in 2025 and revised for ICLR 2026, jointly estimates multiple human bodies, scene geometry, and camera trajectories from monocular video in a common world frame. That makes it particularly relevant to two-person spacing and motion. Its lightweight variant reports approximately 14–16 FPS on an RTX 4090 across the paper's datasets; larger variants are slower. Code and checkpoints are available.

This is a promising research integration, not validated fencing measurement. Compare global motion with visible piste geometry, and do not treat plausible meshes or low 2D reprojection error as proof of accurate depth. [Paper and runtime tables](https://arxiv.org/html/2510.06219v2), [implementation](https://github.com/fanegg/Human3R)

### Sapiens2: an accuracy-focused alternative

Released in April 2026, Sapiens2 targets detailed human analysis at high resolution. Its authors report a four-point pose mAP improvement over the preceding Sapiens generation. Released pose checkpoints start around 400 million parameters, making it a candidate for difficult frames or assisted annotation rather than the initial efficiency baseline. More whole-body landmarks are only useful if the particular wrists, elbows, knees, and feet needed here become more accurate. [Paper](https://arxiv.org/abs/2604.21681), [models and runtime requirements](https://github.com/facebookresearch/sapiens2)

### EfficientTAM and SAM 3.1: mask tracking options

EfficientTAM reports comparable performance to SAM 2 Hiera-B+ with roughly twice the speed on A100 and 2.4 times fewer parameters. It is a video segmentation and mask-tracking component, not a skeletal pose model. Potential roles include propagating selected athlete masks and assisting annotations. [Paper](https://arxiv.org/abs/2411.18933), [implementation](https://github.com/yformer/EfficientTAM)

SAM 3.1 also reduces repeated work across tracked objects. Its approximately sevenfold headline speedup was measured with 128 objects on an H100; that does not predict the gain for two fencers. Consider it if segmentation or interactive selection becomes a bottleneck. [Official release notes](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md)

## Proposed first milestone

The first deliverable should be synchronized comparison videos or a comparison viewer, a small results report, and a model recommendation. Keep the evaluation bounded before committing to broader product changes.

1. **Obtain representative footage.** Start with one or two existing bouts supplied as local paths or accessible links, including an example the original handled well and one it handled poorly. Select about ten short clips covering lunges, footwork, overlap, blur, camera movement, and background people. Expand toward the original review's twenty-clip evaluation if the pilot suggests a useful upgrade.
2. **Correct the comparison's foundations.** Preserve participant identities and video presentation timestamps. Correct overlay alignment and frame synchronization so display bugs do not distort judgments about a model. These are proposed prerequisites, not completed fixes.
3. **Run the three pose candidates.** Compare YOLOv8 nano, YOLO26 medium, and RF-DETR Keypoint on the same clips. Record exact package/checkpoint versions, input resolution, precision, hardware, preprocessing, and settings. Measure model latency separately from complete video-processing time and peak GPU memory.
4. **Measure fencing-relevant errors.** Manually label selected frames and dense segments around important events. Track visible-joint error, missing-joint coverage, identity switches, weapon-side left/right mistakes, temporal jitter, and timing error. Keep unobservable landmarks unknown. A smoother overlay is not sufficient evidence of a more accurate motion estimate.
5. **Evaluate 3D on a few exchanges.** Compare Human3R and Fast SAM 3D Body for orientation, lunging, and recovery. Retain the distinction between observed image evidence and inferred depth. Reference 3D capture would be required to quantify true depth accuracy separately.
6. **Validate one coaching observation.** The suggested first feature is arm-extension timing relative to front-foot landing, accompanied by the relevant replay. Define both events, annotate examples, and have a coach assess usefulness. Viewpoint or occlusion may require an uncertain or unavailable result.

For uploaded videos, a useful later design is to process the full bout with a fast model and reprocess uncertain exchanges with a more expensive model. This is a proposal to evaluate, not an assumption that multiple models will necessarily improve accuracy.

Choose a model from the measured trade-off on these clips. Set acceptable error and processing-time targets with the user before selecting a production winner. If fine-tuning is introduced, split data by bout/source rather than adjacent frames. Check the selected code, checkpoint, and body-model licenses and deployment requirements at that point; the candidates do not all share the same terms.

## What remains open

- Representative footage has been requested but is not present in the reviewed repository inventory.
- Confirm the actual execution machine and available GPU memory. The historical handoff mentions a 4070 Super; its deployment status was not verified during this review.
- No package upgrades, GPU inference, model downloads, benchmark harness, or coaching feature were implemented during the research.
- Gaussian splatting is not the first evaluation target. Body-motion recovery and scene geometry have a more direct proposed role in coaching. Precise blade reconstruction remains a separate task requiring weapon-specific observations and validation.
