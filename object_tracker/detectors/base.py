from abc import ABC, abstractmethod
import numpy as np
from object_tracker.types import Detection
from typing import List

class DetectorBase(ABC):
    @abstractmethod
    def get_detections(self, img: np.ndarray) -> List[Detection]:
        """
        Run detection on an input image and return a list of Detection objects.
        """
        pass
