"""Abstract base class for detector delegates."""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple
import numpy as np


class DetectionResult:
    """Represents a single detection result (uniform for all delegates)."""
    def __init__(self, bbox: Tuple[float, float, float, float], confidence: float, label: str, data: Optional[dict] = None):
        self.bbox = bbox
        self.confidence = confidence
        self.label = label
        self.data = data

    @property
    def location(self):
        # For backward compatibility: treat 'location' as alias for 'label'
        return self.label

    def to_dict(self) -> dict:
        return {
            'bbox': {
                'x1': self.bbox[0],
                'y1': self.bbox[1],
                'x2': self.bbox[2],
                'y2': self.bbox[3]
            },
            'confidence': round(self.confidence, 3),
            'label': self.label,
            'location': self.label,
            'data': self.data
        }


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
