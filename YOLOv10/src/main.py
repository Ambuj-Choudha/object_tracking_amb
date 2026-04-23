import os
import sys
import cv2

from detector import YOLOv10Detector
from detector_onnx import YOLOv10DetectorONNX
from utils.draw_detections import Visualizer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

# Constants
INPUT_FOLDER = "input_images"
FILE_NAME = "horse.jpg"
OUTPUT_FOLDER = "output"

MODEL_WEIGHTS = "yolov10m.pt"
ONNX_FOLDER = "onnx_model"
ONNX_MODEL = os.path.join(BASE_DIR, ONNX_FOLDER, "yolov10m.onnx")

DATASET_FOLDER = "dataset"
DATASET_FILE = os.path.join(PROJECT_ROOT, DATASET_FOLDER, "coco.names")
ONNX = True

if __name__ == "__main__":

    with open(DATASET_FILE, 'r') as file: 
        class_names = file.read().splitlines()
    
    img_path = os.path.join(PROJECT_ROOT, INPUT_FOLDER, FILE_NAME)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")
    
    if ONNX:
        detector = YOLOv10DetectorONNX(ONNX_MODEL)
    else:
        detector = YOLOv10Detector(MODEL_WEIGHTS)

    detections = detector.get_detections(img)

    visualizer = Visualizer(class_names)
    visualizer.draw_detections(img, detections)

    output_path = os.path.join(PROJECT_ROOT, OUTPUT_FOLDER, "detections.jpg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Detection results saved to {output_path}")