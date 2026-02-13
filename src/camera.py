"""Camera capture module for video feed streaming.

This module handles camera initialization and frame capture using OpenCV,
optimized for Nvidia Jetson Orin Nano hardware.
"""

import cv2
from typing import Optional, Tuple
import numpy as np
import time


class CameraCapture:
    """Handles camera initialization and frame capture."""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """Initialize camera capture.
        
        Args:
            camera_id: Camera device ID (default: 0)
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.capture: Optional[cv2.VideoCapture] = None
        
        # FPS tracking
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0
        
    def initialize(self) -> bool:
        """Initialize the camera device.
        
        Returns:
            True if initialization successful, False otherwise
        """
        self.capture = cv2.VideoCapture(self.camera_id)
        
        if not self.capture.isOpened():
            return False
            
        # Set resolution
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Set buffer size to 1 to always get the latest frame (critical for YOLO)
        # This prevents processing stale frames when detection is slower than capture rate
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Enable hardware acceleration for Jetson
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # type: ignore[attr-defined]
        
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the latest frame from the camera.
        
        With CAP_PROP_BUFFERSIZE set to 1 during initialization,
        this always returns the most recent frame, which is critical
        for real-time object detection (YOLO).
        
        Returns:
            Tuple of (success, frame) where success is a boolean
            and frame is a numpy array or None
        """
        if self.capture is None or not self.capture.isOpened():
            return False, None
            
        success, frame = self.capture.read()
        
        # Update FPS calculation
        if success:
            self._frame_count += 1
            current_time = time.time()
            elapsed = current_time - self._fps_start_time
            
            # Calculate FPS once per second
            if elapsed >= 1.0:
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_start_time = current_time
        
        return success, frame
    
    def get_jpeg_frame(self, quality: int = 85) -> Optional[bytes]:
        """Get a JPEG-encoded frame from the camera.
        
        Args:
            quality: JPEG quality (0-100)
            
        Returns:
            JPEG-encoded frame as bytes or None if capture fails
        """
        success, frame = self.read_frame()
        
        if not success or frame is None:
            return None
            
        # Encode frame as JPEG
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode('.jpg', frame, encode_params)
        
        if not success:
            return None
            
        return buffer.tobytes()
    
    def get_fps(self) -> float:
        """Get the current frames per second.
        
        FPS is calculated once per second based on actual frame capture rate.
        
        Returns:
            Current FPS as a float, or 0.0 if not yet calculated
        """
        return self._current_fps
    
    def release(self) -> None:
        """Release the camera device."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
    
    def is_opened(self) -> bool:
        """Check if camera is opened.
        
        Returns:
            True if camera is opened, False otherwise
        """
        return self.capture is not None and self.capture.isOpened()
