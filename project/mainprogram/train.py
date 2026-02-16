from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(
    data="project/databaseMAIN/data.yaml",
    epochs=50,        
    imgsz=640,        
    batch=32,         
    device=0,
    workers=0       
)
