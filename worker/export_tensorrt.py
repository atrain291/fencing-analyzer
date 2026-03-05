"""Export YOLO11x-Pose to TensorRT engine format.

Run once inside the worker container to generate the engine file:

    podman exec -it fencing-analyzer_worker_1 python export_tensorrt.py

The resulting yolo11x-pose.engine file is tied to:
  - GPU architecture (SM 8.9 for RTX 4070 Super)
  - TensorRT version installed in the container
  - Input resolution (imgsz=1280)

If any of these change, re-run this script to regenerate the engine.
The engine file will be saved next to the .pt weights at /app/yolo11x-pose.engine.
Since /app is bind-mounted from ./worker, the engine persists on the host.
"""
from ultralytics import YOLO

MODEL_PATH = "yolo11x-pose.pt"

print(f"Loading {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

print("Exporting to TensorRT (FP16, imgsz=1280, batch=1)...")
print("This may take several minutes on first run.")
engine_path = model.export(
    format="engine",
    imgsz=1280,
    half=True,       # FP16 — best speed/accuracy balance for pose on RTX 4070 Super
    dynamic=False,   # Fixed input size is faster than dynamic
    batch=1,
    device="0",
)

print(f"Export complete! Engine saved to: {engine_path}")
print("Restart the worker to use the TensorRT engine automatically.")
