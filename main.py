import os

import cv2
import numpy as np


def preprocess_images(padding, img_size, resize_method, img):
    pass

def draw_label(img, colour, obj_width, obj_height, cx, cy, thickness):
    corner1_x = cx - obj_width // 2
    corner1_y = cy + obj_height // 2

    corner2_x = cx + obj_width // 2
    corner2_y = cy - obj_height // 2

    corner_point1 = (corner1_x, corner1_y)
    corner_point2 = (corner2_x, corner2_y)

    cv2.rectangle(img, corner_point1, corner_point2, colour, thickness)
    pass


if __name__ == "__main__":
    # Constants
    INPUT_WIDTH = 416
    INPUT_HEIGHT = 416
    COLOUR = (45, 35, 56)
    THICKNESS = 2
    INPUT_FOLDER = "input_images"
    FILE_NAME = "horse.jpg"

    img_path = os.path.join(INPUT_FOLDER, FILE_NAME)
    img = cv2.imread(img_path)

    # A blob is a 4D numpy array object (images, channels, width, height)
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), 
                                swapRB=True, crop=False)

    # load trained model
    model = cv2.dnn.readNetFromDarknet('model_config/yolov3.cfg', 'yolov3.weights')
    model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    
    layers = model.getLayerNames()
    output_layer = [layers[i - 1] for i in model.getUnconnectedOutLayers()]

    # pass the processed image (blob) and get inference
    model.setInput(blob)
    predictions = model.forward(output_layer)  

    # visualise the output
    for prediction in predictions:    # predictions are stacked for different scales
        for detection in prediction:  # For each detection there is a vector: cx, cy, w, h, obectjness score + 80 classes
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            objectness = detection[4]

            if objectness > 0.5:
                center_x = int(detection[0] * img.shape[1])  # cx and cy are scaled to image dimensions
                center_y = int(detection[1] * img.shape[0])
                width = int(detection[2] * img.shape[1])
                height = int(detection[3] * img.shape[0])
                draw_label(img, COLOUR, width, height, center_x, center_y, THICKNESS)
    
    # show final image
    cv2.imshow("Detections", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()