
from ultralytics import YOLO

checkpoint = (
    r"runs\segment\runs\crater_rock\yolov8s_seg\weights\last.pt"
)

model = YOLO(checkpoint)
model.train(resume=True)
