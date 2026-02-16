"""Camera capture module for video feed streaming.

This module handles camera initialization and frame capture using OpenCV,
optimized for Nvidia Jetson Orin Nano hardware.
"""


import cv2
from typing import Optional, Tuple
import numpy as np
import time
from .camera_base import CameraBase


class CameraCapture(CameraBase):
    """Handles camera initialization and frame capture."""
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        super().__init__()
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.capture: Optional[cv2.VideoCapture] = None
        # FPS tracking
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0

    def initialize(self) -> bool:
        self.capture = cv2.VideoCapture(self.camera_id)
        if not self.capture.isOpened():
            return False
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # type: ignore[attr-defined]
        return True

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the latest frame from the camera.
        Returns:
            Tuple (success, frame) where frame is a numpy array
        """
        if self.capture is None:
            return False, None
        success, frame = self.capture.read()
        # FPS tracking (optional, not used in base)
        self._frame_count += 1
        if time.time() - self._fps_start_time > 1.0:
            self._current_fps = self._frame_count / (time.time() - self._fps_start_time)
            self._frame_count = 0
            self._fps_start_time = time.time()
        return success, frame

    def is_opened(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def get_fps(self) -> float:
        """Get the current frames per second.
        FPS is calculated once per second based on actual frame capture rate.
        Returns:
            Current FPS as a float, or 0.0 if not yet calculated
        """
        return self._current_fps
