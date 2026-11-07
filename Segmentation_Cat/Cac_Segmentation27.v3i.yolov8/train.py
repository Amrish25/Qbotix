
from ultralytics import YOLO
import torch
from pathlib import Path

# Verify CUDA
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU not detected!")

print("GPU:", torch.cuda.get_device_name(0))

# Dataset configuration
DATA = r"C:\Users\AMRISH A\Downloads\crater_rock_seg\data.yaml"

if not Path(DATA).exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA}")

# Load YOLOv8 small segmentation model
model = YOLO("yolov8s-seg.pt")

# Train
results = model.train(
    data=DATA,
    epochs=150,
    imgsz=640,
    batch=8,
    device=0,
    workers=0,

    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    weight_decay=0.0005,

    patience=30,
    cos_lr=True,
    amp=True,

    project="runs/crater_rock",
    name="yolov8s_seg",
    exist_ok=True,

    save=True,
    save_period=10,
    plots=True,
    verbose=True
)

print("\nTraining completed!")
print("Best model:", Path("runs/crater_rock/yolov8s_seg/weights/best.pt").resolve())
