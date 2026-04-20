
import numpy as np
import cv2


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
        corner1_x = int(round(cx - w / 2))
        corner1_y = int(round(cy - h / 2))

        # corner 2: bottom-right corner
        corner2_x = int(round(cx + w / 2))
        corner2_y = int(round(cy + h / 2))

        top_left = (corner1_x, corner1_y)
        bottom_right = (corner2_x, corner2_y) 

        cv2.rectangle(img, top_left, bottom_right, colour, self.thickness)

        # write the labels on top of the bbox
        
        class_name = self.class_names[detection.class_id]
        y = max(cy - (h // 2) - 5, 10)  # just below the top edge, clamped to frame  
        
        cv2.putText(img, class_name, (int(round(cx)), int(round(y))), fontFace=self.font, fontScale=0.5, 
                    thickness=self.thickness, color=self.text_colour, bottomLeftOrigin=False)

    @staticmethod
    def _generate_colours(n):
        np.random.seed(42)  # deterministic colours across runs
        return [tuple(int(c) for c in colour) 
                for colour in np.random.randint(0, 255, (n, 3))]
