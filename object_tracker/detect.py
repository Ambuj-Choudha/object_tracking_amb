import argparse
import logging
import os

import cv2

import object_tracker.config
from object_tracker.detectors.yolov10 import YOLOv10DetectorONNX
from object_tracker.visualization.draw_detections import Visualizer

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-image",    required=True,                  help="Input image filename (inside DATA/INPUT/)")
    parser.add_argument("--output-folder",  default=object_tracker.config.OUTPUT_FOLDER,   help="Folder to save results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with open(object_tracker.config.DATASET_FILE, 'r') as file: 
        class_names = file.read().splitlines()
    
    img_path = os.path.join(object_tracker.config.INPUT_FOLDER, args.input_image)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {img_path}")
    
    detector = YOLOv10DetectorONNX(object_tracker.config.ONNX_MODEL, object_tracker.config.CONFIDENCE_THRESHOLD)

    detections = detector.get_detections(img)

    visualizer = Visualizer(class_names)
    visualizer.draw_detections(img, detections)

    name, ext = os.path.splitext(args.input_image)
    output_path = os.path.join(args.output_folder, f"{name}_detections{ext}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    logger.info("Detection results saved to %s", output_path)


if __name__ == "__main__":
    main()
