"""Camera capture module for video feed streaming.

This module handles camera initialization and frame capture using OpenCV,
optimized for Nvidia Jetson Orin Nano hardware.
"""

import cv2
from typing import Optional, Tuple
import numpy as np


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
        
        # Enable hardware acceleration for Jetson
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # type: ignore[attr-defined]
        
        return True
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the most recent frame from the camera.
        
        This method flushes the camera buffer to ensure we get the latest
        frame, which is critical for real-time object detection (YOLO).
        This prevents processing stale frames when detection is slower
        than the camera's capture rate.
        
        Returns:
            Tuple of (success, frame) where success is a boolean
            and frame is a numpy array or None
        """
        if self.capture is None or not self.capture.isOpened():
            return False, None
        
        # Flush buffer by grabbing frames without decoding (fast operation)
        # This ensures we get the most recent frame, not a buffered old one
        for _ in range(4):
            self.capture.grab()
            
        # Now read and decode the latest frame
        success, frame = self.capture.read()
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
