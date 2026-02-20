from typing import Tuple, Optional


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