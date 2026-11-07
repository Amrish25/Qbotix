
from ultralytics import YOLO
from pathlib import Path
import torch

DATA = r"C:\Users\AMRISH A\Downloads\crater_rock_seg\data.yaml"

MODEL = r"runs\segment\runs\crater_rock\yolov8s_seg\weights\best.pt"

if not Path(DATA).exists():
    raise FileNotFoundError(f"Dataset not found: {DATA}")

if not Path(MODEL).exists():
    raise FileNotFoundError(f"Model not found: {MODEL}")

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not available")

print("GPU:", torch.cuda.get_device_name(0))

model = YOLO(MODEL)

print("\nEvaluating on test dataset...")

metrics = model.val(
    data=DATA,
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    plots=True,
    project="runs/evaluation",
    name="crater_rock_test",
    exist_ok=True
)

print("\n========== TEST RESULTS ==========")
print("Mask Precision:", metrics.seg.mp)
print("Mask Recall:", metrics.seg.mr)
print("Mask mAP50:", metrics.seg.map50)
print("Mask mAP50-95:", metrics.seg.map)

print("\nPer-class Mask mAP50-95:")
for class_id, score in enumerate(metrics.seg.maps):
    print(f"{metrics.names[class_id]}: {score:.4f}")

print("\nEvaluation completed!")
