import argparse
import cv2
import config
from detector import YOLOv10Detector
from detector_onnx import YOLOv10DetectorONNX
import os
from utils.draw_detections import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image",    required=True,                  help="Input image filename (inside INPUT_FOLDER/)")
    parser.add_argument("--output-folder",  default=config.OUTPUT_FOLDER,   help="Folder to save results")
    parser.add_argument("--onnx-inference", action="store_true",            help="To use ONNX runtime")
    args = parser.parse_args()


    with open(config.DATASET_FILE, 'r') as file: 
        class_names = file.read().splitlines()
    
    img_path = os.path.join(config.INPUT_FOLDER, args.input_image)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")
    
    if args.onnx_inference:
        detector = YOLOv10DetectorONNX(config.ONNX_MODEL, config.CONFIDENCE_THRESHOLD)
    else:
        detector = YOLOv10Detector(config.PYTORCH_MODEL_WEIGHTS, config.CONFIDENCE_THRESHOLD, config.NMS_IOU_THRESHOLD)

    detections = detector.get_detections(img)

    visualizer = Visualizer(class_names)
    visualizer.draw_detections(img, detections)

    name, ext = os.path.splitext(args.input_image)
    output_path = os.path.join(args.output_folder, f"{name}_detections{ext}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Detection results saved to {output_path}")
