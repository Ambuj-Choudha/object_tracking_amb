from abc import ABC, abstractmethod
import numpy as np
from object_tracker.types import Detection

class DetectorBase(ABC):
    @abstractmethod
    def get_detections(self, img: np.ndarray) -> list[Detection]:
        """
        Run detection on an input image and return a list of Detection objects.
        """
        pass
