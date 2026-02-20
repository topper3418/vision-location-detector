"""Example processor implementation for location calculation and analytics."""
from src.processor import DetectionProcessor
from src.video_feed_base import DetectionResult
from typing import List
import numpy as np

class LocationAnalyticsProcessor(DetectionProcessor):
    """Processes detections to add location info and perform analytics."""
    def process(self, frame: np.ndarray, detections: List[DetectionResult]) -> None:
        height, width = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            # Horizontal position
            if center_x < width / 3:
                horizontal = "Left"
            elif center_x < 2 * width / 3:
                horizontal = "Center"
            else:
                horizontal = "Right"
            # Depth/distance
            if center_y > 2 * height / 3:
                depth = "Near"
            elif center_y > height / 3:
                depth = "Mid"
            else:
                depth = "Far"
            det.label = f"{horizontal}-{depth}"  # Overwrite label for demo
        # Additional analytics or alerts can be added here
