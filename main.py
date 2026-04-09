import os

import cv2
import numpy as np

from detector import YOLODetector
from detector import Visualizer


if __name__ == "__main__":
    # Constants
    INPUT_FOLDER = "input_images"
    FILE_NAME = "horse.jpg"
    MODEL_CONFIG = 'model_config/yolov3.cfg'
    MODEL_WEIGHTS = 'yolov3.weights'
    DATASET_FILE = 'coco.names'

    # get classes for the dataset
    with open(DATASET_FILE, 'r') as file: 
        class_names = file.read().splitlines()
    
    # read the input image
    img_path = os.path.join(INPUT_FOLDER, FILE_NAME)
    img = cv2.imread(img_path)

    # load trained model
    detector = YOLODetector(MODEL_CONFIG, MODEL_WEIGHTS)

    # pass the processed image and get inference
    detections = detector.get_detections(img)

    # visualise the output
    visualizer = Visualizer(class_names)
    visualizer.draw_detections(img, detections)

    # show final image
    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()