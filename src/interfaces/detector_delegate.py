"""Abstract base class for detector delegates."""
from abc import ABC, abstractmethod
from typing import Any, List, Optional
import numpy as np

from src.interfaces.detection_result import DetectionResult


class DetectorDelegate(ABC):
    """Base class for all detector delegates."""

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the detector. Returns True on success."""
        pass

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Run detection on a frame. Returns detection results."""
        pass

    @abstractmethod
    def draw_detections(self, frame: np.ndarray, detections: Any) -> np.ndarray:
        """Draw detections on a frame. Returns annotated frame."""
        pass

    @abstractmethod
    def release(self) -> None:
        """Release any resources held by the detector."""
        pass
