# config.py
import os

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# Model assets
ONNX_MODEL = os.path.join(PROJECT_ROOT, "assets", "models", "yolov10m.onnx")

# Detection
CONFIDENCE_THRESHOLD = 0.5
# NMS_IOU_THRESHOLD = 0.7  # not needed anymore YOLOv10 is nms-free

# Dataset
DATASET_FILE = os.path.join(PROJECT_ROOT, "assets", "labels", "coco.names")

# I/O
INPUT_FOLDER  = os.path.join(PROJECT_ROOT, "data", "input")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "data", "output")  # can be overidden by user
