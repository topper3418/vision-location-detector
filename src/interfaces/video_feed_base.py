
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
        self._fps_frame_count = 0
        self._fps_start_time = None

    @property
    def measure_fps(self) -> bool:
        """Whether to measure FPS during frame reading."""
        return self._measure_fps

    @measure_fps.setter
    def measure_fps(self, value: bool):
        self._measure_fps = value
        if value:
            self._fps_frame_count = 0
            import time
            self._fps_start_time = time.time()
        else:
            self._fps_start_time = None

    @property
    def fps(self) -> float:
        """Return the most recently calculated FPS value."""
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
        """Yields raw frames from the video feed. If measure_fps is enabled, calculates FPS after each frame."""
        import time
        while True:
            success, frame = self.read_frame()
            if not success or frame is None:
                break
            if self._measure_fps:
                self._fps_frame_count += 1
                now = time.time()
                if self._fps_start_time is not None:
                    elapsed = now - self._fps_start_time
                    if elapsed > 0:
                        self._last_fps = self._fps_frame_count / elapsed
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
