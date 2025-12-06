from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')
model.val(data='./data/data.yaml', batch=4, workers=0)