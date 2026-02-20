
"""Video feed base class and detection delegate ABCs."""
from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple, Generator
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.interfaces.detection_result import DetectionResult
from src.interfaces.detector_delegate import DetectorDelegate


class VideoFeedBase(ABC):
    """Abstract video feed interface supporting delegate-based detection and postprocessing pipeline."""
    def __init__(self):
        self.detection_delegate = None
        self._measure_fps = False
        self._last_fps = 0.0
        self._fps_timestamps = []  # List of frame timestamps for sliding window FPS
        self._fps_window_seconds = 1.0  # Window size in seconds for FPS calculation

    @property
    def measure_fps(self) -> bool:
        """Whether to measure FPS during frame reading."""
        return self._measure_fps

    @measure_fps.setter
    def measure_fps(self, value: bool):
        self._measure_fps = value
        import time
        if value:
            self._fps_timestamps = []
        else:
            self._fps_timestamps = []

    @property
    def fps(self) -> float:
        """Return the most recently calculated FPS value (over the last second)."""
        return self._last_fps

    @abstractmethod
    def initialize(self) -> bool:
        pass

    @abstractmethod
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        pass

    @abstractmethod
    def is_opened(self) -> bool:
        pass

    @abstractmethod
    def release(self):
        pass

    def set_detector_delegate(self, delegate: DetectorDelegate):
        self.detection_delegate = delegate

    def get_raw_stream(self) -> Generator[np.ndarray, None, None]:
        """Yields raw frames from the video feed. If measure_fps is enabled, calculates FPS using a sliding window."""
        import time
        while True:
            success, frame = self.read_frame()
            if not success or frame is None:
                break
            if self._measure_fps:
                now = time.time()
                self._fps_timestamps.append(now)
                # Remove timestamps outside the window
                window_start = now - self._fps_window_seconds
                self._fps_timestamps = [t for t in self._fps_timestamps if t >= window_start]
                count = len(self._fps_timestamps)
                if count > 1:
                    duration = self._fps_timestamps[-1] - self._fps_timestamps[0]
                    if duration > 0:
                        self._last_fps = (count - 1) / duration
                    else:
                        self._last_fps = 0.0
                else:
                    self._last_fps = 0.0
            yield frame

    def get_full_stream(self) -> Generator[Tuple[Optional[List[DetectionResult]], np.ndarray], None, None]:
        """Yields (detections, processed_frame) tuples using the delegate and postprocessors."""
        for frame in self.get_raw_stream():
            detections = None
            processed = frame
            if self.detection_delegate:
                detections = self.detection_delegate.detect(frame)
                processed = self.detection_delegate.draw_detections(frame, detections)
            yield (detections, processed)

    def get_processed_stream(self) -> Generator[Tuple[Optional[List[DetectionResult]], np.ndarray], None, None]:
        """Yields processed frames (e.g., annotated) from the video feed."""
        for frame in self.get_raw_stream():
            processed = frame
            detections = None
            if self.detection_delegate:
                detections = self.detection_delegate.detect(frame)
                processed = self.detection_delegate.draw_detections(frame, detections)
            yield (detections, processed)

    def get_data_stream(self) -> Generator[Optional[List[DetectionResult]], None, None]:
        """Yields only the detection results from the delegate (headless mode)."""
        for frame in self.get_raw_stream():
            if self.detection_delegate:
                detections = self.detection_delegate.detect(frame)
                yield detections
            else:
                yield None
