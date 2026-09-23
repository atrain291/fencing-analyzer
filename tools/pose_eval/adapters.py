"""Lazy optional model APIs with one COCO-17 output schema and shared tracking."""
from pathlib import Path
from time import perf_counter

from worker.app.pipeline.pose import KEYPOINT_NAMES
from . import MODEL_IDS
from .runner import sha256_file


def normalize_predictions(boxes, scores, coordinates, confidences, width, height, covariance=None):
    """Keep unobserved/nonfinite joints absent; coordinates are normalized upright pixels."""
    import numpy as np

    if len(boxes) != len(scores) or len(boxes) != len(coordinates) or len(boxes) != len(confidences):
        raise ValueError("Model box/keypoint counts differ")
    detections = []
    for index, (box, score, points, confidence) in enumerate(zip(boxes, scores, coordinates, confidences)):
        if len(box) != 4 or not np.isfinite(box).all() or not np.isfinite(score):
            continue
        if len(points) != 17 or len(confidence) != 17:
            raise ValueError("This evaluation requires the pretrained COCO 17-keypoint skeleton")
        joints = {}
        for joint_index, (name, point, certainty) in enumerate(zip(KEYPOINT_NAMES, points, confidence)):
            if len(point) != 2 or not np.isfinite(point).all() or not np.isfinite(certainty) or certainty <= 0:
                continue
            x, y = float(point[0] / width), float(point[1] / height)
            if not (0 <= x <= 1 and 0 <= y <= 1):
                continue
            joint = {"x": x, "y": y, "confidence": float(certainty)}
            if covariance is not None:
                matrix = np.asarray(covariance[index][joint_index])
                if matrix.shape == (2, 2) and np.isfinite(matrix).all():
                    scale = np.asarray([width, height])
                    joint["covariance"] = (matrix / np.outer(scale, scale)).tolist()
            joints[name] = joint
        normalized_box = (np.asarray(box) / [width, height, width, height]).tolist()
        detections.append({"box": normalized_box, "confidence": float(score), "track_id": None, "keypoints": joints})
    return detections


class ModelAdapter:
    def __init__(self, model_id, *, device="cuda:0", image_size=None, threshold=0.25, weights=None):
        if model_id not in MODEL_IDS:
            raise ValueError(f"Unknown model: {model_id}")
        if not 0 < threshold < 1:
            raise ValueError("Detection threshold must be between zero and one")
        if image_size is not None and (type(image_size) is not int or image_size <= 0):
            raise ValueError("Image size must be a positive integer")
        if weights is not None and not Path(weights).is_file():
            raise ValueError(f"Checkpoint not found: {weights}")
        try:
            import torch
            import supervision as sv
        except ImportError as exc:
            raise RuntimeError("Install the optional evaluation model requirements in a separate environment") from exc
        self.torch, self.sv = torch, sv
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Evaluation currently supports explicit cpu or cuda devices")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu for a smoke run, or run on the GPU host")
        self.model_id, self.threshold = model_id, threshold
        self.image_size = image_size or (576 if model_id == "rfdetr-keypoint" else 640)
        began = perf_counter()
        try:
            if model_id == "rfdetr-keypoint":
                import rfdetr

                model_class = getattr(rfdetr, "RFDETRKeypointPreview", None)
                if model_class is None:
                    raise RuntimeError("Installed rfdetr lacks RFDETRKeypointPreview; use the documented evaluation version")
                kwargs = {"device": str(self.device), "resolution": self.image_size}
                if weights:
                    kwargs["pretrain_weights"] = str(Path(weights).resolve())
                self.model = model_class(**kwargs)
                checkpoint = self.model.model_config.pretrain_weights
            else:
                from ultralytics import YOLO

                checkpoint = str(weights or f"{model_id}-pose.pt")
                self.model = YOLO(checkpoint)
                checkpoint = getattr(self.model, "ckpt_path", checkpoint)
        except ImportError as exc:
            raise RuntimeError(f"Missing {model_id} dependencies; see evaluation/README.md") from exc
        checkpoint_path = Path(checkpoint) if checkpoint else None
        self.metadata = {
            "model_id": model_id, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path and checkpoint_path.is_file() else None,
            "device": str(self.device), "image_size": self.image_size, "precision": "float32",
            "detection_threshold": threshold, "setup_seconds": perf_counter() - began,
            "tracker": "supervision.ByteTrack", "tracker_frame_rate_setting": 30,
            "tracker_note": "Same frame-based tracker/settings for all models; VFR timestamps remain unchanged.",
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "cuda_runtime": torch.version.cuda, "synthetic": False,
        }
        self.reset()

    def reset(self):
        self.tracker = self.sv.ByteTrack(track_activation_threshold=self.threshold)

    def synchronize(self):
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def reset_memory_stats(self):
        if self.device.type == "cuda":
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def peak_memory(self):
        return self.torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else None

    def predict(self, image):
        import numpy as np

        height, width = image.shape[:2]
        with self.torch.inference_mode():
            if self.model_id == "rfdetr-keypoint":
                result = self.model.predict(np.ascontiguousarray(image[:, :, ::-1]), threshold=self.threshold)
                if len(result.xy) == 0:
                    return []
                if result.keypoint_confidence is None or result.detection_confidence is None or "xyxy" not in result.data:
                    raise RuntimeError("RF-DETR output lacks required keypoint/box/confidence fields")
                return normalize_predictions(result.data["xyxy"], result.detection_confidence, result.xy,
                                             result.keypoint_confidence, width, height, result.data.get("covariance"))
            result = self.model.predict(image, device=str(self.device), imgsz=self.image_size,
                                        conf=self.threshold, half=False, verbose=False, classes=[0])[0]
            if result.keypoints is None or len(result.keypoints) == 0:
                return []
            if result.keypoints.conf is None:
                raise RuntimeError("YOLO pose output has no per-keypoint confidence")
            return normalize_predictions(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                         result.keypoints.xy.cpu().numpy(), result.keypoints.conf.cpu().numpy(), width, height)

    def track(self, detections, width, height):
        import numpy as np

        boxes = np.asarray([d["box"] for d in detections], dtype=np.float32).reshape(-1, 4)
        boxes *= [width, height, width, height]
        inputs = self.sv.Detections(xyxy=boxes, confidence=np.asarray([d["confidence"] for d in detections]),
                                    class_id=np.zeros(len(detections), dtype=int),
                                    data={"source_index": np.arange(len(detections))})
        tracked = self.tracker.update_with_detections(inputs)
        # Preserve every detection, including observations not yet activated by ByteTrack.
        if len(tracked):
            for source_index, track_id in zip(tracked.data["source_index"], tracked.tracker_id):
                detections[int(source_index)]["track_id"] = int(track_id)
        return detections
