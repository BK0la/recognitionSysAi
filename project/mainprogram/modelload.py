import os, sys
from ultralytics import YOLO

def resource_path(rel_path: str) -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.dirname(__file__)) 
    return os.path.join(base, rel_path)

def load_model():
    weights = resource_path(os.path.join("weights", "best.pt"))
    return YOLO(weights)
