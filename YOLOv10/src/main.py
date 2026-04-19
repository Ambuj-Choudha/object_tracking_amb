import os
import sys
import cv2

from detector import YOLOv10Detector
from common_utils.visualizer import Visualizer


if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

    # Constants
    INPUT_FOLDER = "input_images"
    FILE_NAME = "horse.jpg"
    MODEL_WEIGHTS = 'yolov10m.pt'
    DATASET_FOLDER = 'dataset'
    DATASET_FILE = os.path.join(PROJECT_ROOT, DATASET_FOLDER, 'coco.names')

    # get classes for the dataset
    with open(DATASET_FILE, 'r') as file: 
        class_names = file.read().splitlines()
    
    # read the input image
    img_path = os.path.join(PROJECT_ROOT, INPUT_FOLDER, FILE_NAME)
    img = cv2.imread(img_path)

    # load trained model
    detector = YOLOv10Detector(MODEL_WEIGHTS)

    # pass the processed image and get inference
    detections = detector.get_detections(img)

    # visualise the output
    visualizer = Visualizer(class_names)
    visualizer.draw_detections(img, detections)

    # show final image
    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()