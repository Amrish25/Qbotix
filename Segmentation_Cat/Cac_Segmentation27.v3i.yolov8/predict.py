
from ultralytics import YOLO

model = YOLO(
    r"runs\segment\runs\crater_rock\yolov8s_seg\weights\best.pt"
)

results = model.predict(
    source=r"C:\Users\AMRISH A\Downloads\crater_rock_seg\test\images",
    imgsz=640,
    conf=0.25,
    iou=0.7,
    device=0,
    save=True,
    show_labels=True,
    show_conf=True,
    project="runs/predictions",
    name="crater_rock_test",
    exist_ok=True
)

print("Prediction completed!")
print("Images saved in runs/predictions/crater_rock_test")
