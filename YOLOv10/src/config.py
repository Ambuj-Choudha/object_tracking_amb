# config.py
import os

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

# Model assets
PYTORCH_MODEL_WEIGHTS = os.path.join(BASE_DIR, "yolov10m.pt")
ONNX_MODEL            = os.path.join(BASE_DIR, "onnx_model", "yolov10m.onnx")

# Detection
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.7

# Dataset
DATASET_FILE = os.path.join(PROJECT_ROOT, "dataset", "coco.names")

# I/O
INPUT_FOLDER  = os.path.join(PROJECT_ROOT, "input_images")
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "output")  # can be overidden by user
