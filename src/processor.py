"""Detection post-processing interface for server-side logic."""
from abc import ABC, abstractmethod
from typing import List
from src.video_feed_base import DetectionResult
import numpy as np

class DetectionProcessor(ABC):
    """Abstract base class for processing detection results."""
    @abstractmethod
    def process(self, frame: np.ndarray, detections: List[DetectionResult]) -> None:
        """Process detection results (e.g., tracking, analytics, alerts).
        Args:
            frame: The video frame (np.ndarray)
            detections: List of DetectionResult objects
        """
        pass
