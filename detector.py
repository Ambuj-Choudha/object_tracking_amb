from dataclasses import dataclass

import numpy as np
import cv2


class YOLODetector():
    def __init__(self, config, weights):
        self.objectness_threshold = 0.5
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

        self._scale = 1/255.0
        self._input_size = (416, 416)

        self.model = cv2.dnn.readNetFromDarknet(config, weights)
        self.model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        layers = self.model.getLayerNames()
        self.output_layer = [layers[i - 1] for i in self.model.getUnconnectedOutLayers()]

    # interface for the user, images ---> detections
    def get_detections(self, img):
        blob = self._preprocess(img)
        raw_predictions = self._predict(blob)
        detections = self._postprocess(raw_predictions, img.shape)
        return detections

    # private methods
    def _predict(self, blob):
        self.model.setInput(blob)
        predictions = self.model.forward(self.output_layer)
        return predictions
        
    def _preprocess(self, img):
        blob = cv2.dnn.blobFromImage(img, self._scale, self._input_size, swapRB=True, crop=False)
        return blob
    
    # convert the detector's output into standard Data Structure for detections
    def _postprocess(self, predictions, img_shape):
        image_height, image_width = img_shape[:2]
        detections = []

        for prediction in predictions:    # predictions are stacked for different scales
            for detection in prediction:  # For each detection there is a vector: cx, cy, w, h, obectjness score + 80 classes
                objectness = detection[4]
                
                if objectness > self.objectness_threshold:
                    scores = detection[5:]

                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
            
                    center_x = int(detection[0] * image_width)  # cx and cy are scaled to image dimensions
                    center_y = int(detection[1] * image_height)

                    width = int(detection[2] * image_width)
                    height = int(detection[3] * image_height)

                    bbox = (center_x, center_y, width, height)

                    detections.append(Detection(class_id, confidence, bbox))
            
        detections_after_nms = self._apply_nms(detections, self.nms_threshold)
        return detections_after_nms
    
    def _apply_nms(self, detections, nms_threshold):
        # cv2.dnn.NMSBoxes expects (x, y, w, h) - convert from center format
        bboxes = [(cx - w // 2, cy - h // 2, w, h) for (cx, cy, w, h) in (d.bbox for d in detections)]
        
        confidences = [d.confidence for d in detections]

        indices = cv2.dnn.NMSBoxes(bboxes, confidences, self.threshold, nms_threshold)  # indices of filtered bbox

        filtered_detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                filtered_detections.append(detections[i])

        return filtered_detections

@dataclass
class Detection:
    class_id: int
    confidence: float
    bbox: tuple  # (cx, cy, w, h)


class Visualizer():
    def __init__(self, class_names, colour_map=None, text_colour=(255, 255, 255), thickness=2):
        self.class_names = class_names
        self.thickness = thickness
        self.text_colour = text_colour
        self.colour_map = colour_map or self._generate_colours(len(class_names))
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    # interface for the user
    def draw_detections(self, img, detections):
        for detection in detections:
            self._draw_bbox_w_labels(img, detection)

    # private methods
    def _draw_bbox_w_labels(self, img, detection):
        colour = self.colour_map[detection.class_id]
        cx, cy, w, h = detection.bbox

        # corner 1: top-left corner
        corner1_x = cx - w // 2
        corner1_y = cy + h // 2

        # corner 2: bottom-right corner
        corner2_x = cx + w // 2
        corner2_y = cy - h // 2

        top_left = (corner1_x, corner1_y)
        bottom_right = (corner2_x, corner2_y) 

        cv2.rectangle(img, top_left, bottom_right, colour, self.thickness)

        # write the labels on top of the bbox
        class_name = self.class_names[detection.class_id]
        y = max(cy - (h // 2) - 5, 10)  # just below the top edge, clamped to frame  
        
        cv2.putText(img, class_name, (cx, y), fontFace=self.font, fontScale=0.5, 
                    thickness=self.thickness, color=self.text_colour, bottomLeftOrigin=False)

    @staticmethod
    def _generate_colours(n):
        np.random.seed(42)  # deterministic colours across runs
        return [tuple(int(c) for c in colour) 
                for colour in np.random.randint(0, 255, (n, 3))]


