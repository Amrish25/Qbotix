
from pathlib import Path
from ultralytics import YOLO

model = YOLO(
    r"runs\segment\runs\crater_rock\yolov8s_seg\weights\best.pt"
)

image_dir = Path(
    r"C:\Users\AMRISH A\Downloads\crater_rock_seg\test\images"
)

label_dir = Path(
    r"C:\Users\AMRISH A\Downloads\crater_rock_seg\test\labels"
)

images = sorted(image_dir.glob("frame_*.jpg"))

print("Frame images:", len(images))

for image in images:
    label = label_dir / f"{image.stem}.txt"

    gt_count = 0
    if label.exists():
        gt_count = sum(
            1 for line in label.read_text().splitlines()
            if line.strip()
        )

    result = model.predict(
        source=str(image),
        imgsz=640,
        conf=0.25,
        device=0,
        verbose=False
    )[0]

    pred_count = len(result.boxes)

    print(
        f"{image.name} | "
        f"Ground truth: {gt_count} | "
        f"Predictions: {pred_count}"
    )
