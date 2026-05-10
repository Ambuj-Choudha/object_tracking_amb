import numpy as np
import cv2


class Visualizer():
    def __init__(self, class_names, colour_map=None, text_colour=(0, 0, 0), thickness=2):
        self.class_names = class_names
        self.thickness = thickness
        self.text_thickness = max(1, self.thickness // 2)
        self.text_colour = text_colour
        self.colour_map = colour_map or self._generate_colours(len(class_names))
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5

    # interface for the user
    def draw_detections(self, img, detections):
        for detection in detections:
            self._draw_bbox_w_labels(img, detection)

    # private methods
    def _draw_bbox_w_labels(self, img, detection):
        colour = self.colour_map[detection.class_id]
        cx, cy, w, h = detection.bbox
        img_h, img_w = img.shape[:2]

        x1 = max(0, min(int(round(cx - w / 2)), img_w - 1))
        y1 = max(0, min(int(round(cy - h / 2)), img_h - 1))
        x2 = max(0, min(int(round(cx + w / 2)), img_w - 1))
        y2 = max(0, min(int(round(cy + h / 2)), img_h - 1))

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, self.thickness)

        # Label for Bbox text with class name and score
        label = f"{self.class_names[detection.class_id]}: {detection.confidence:.2f}"
        pad = 4

        (label_width, label_height), baseline = cv2.getTextSize(label, self.font, fontScale=self.font_scale, 
                                                                thickness=self.text_thickness)

        # Calculate the position of the label text
        if y1 - label_height - 2 * pad >= 0:
            bg_y1 = y1 - label_height - 2 * pad
            bg_y2 = y1
            text_y = y1 - pad - baseline
        else:
            bg_y1 = y1
            bg_y2 = y1 + label_height + 2 * pad
            text_y = y1 + label_height + pad - baseline

        bg_x1 = x1
        bg_x2 = min(x1 + label_width + 2 * pad, img_w)
        text_x = bg_x1 + pad

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), colour, cv2.FILLED)
        
        # Put the label text on the image
        cv2.putText(img, label, (text_x, text_y), fontFace=self.font, fontScale=self.font_scale, 
                    thickness=self.text_thickness, color=self.text_colour, lineType=cv2.LINE_AA, bottomLeftOrigin=False)

    @staticmethod
    def _generate_colours(n):
        np.random.seed(42)  # deterministic colours across runs
        return [tuple(int(c) for c in colour) 
                for colour in np.random.randint(0, 255, (n, 3))]
