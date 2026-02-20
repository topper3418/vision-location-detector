
"""Spoof video feed class for testing detection at exact 30 FPS."""

import cv2
import time
from ..interfaces.video_feed_base import VideoFeedBase

from ..interfaces.video_feed_base import DetectorDelegate

class SpoofVideoFeed(VideoFeedBase):
    """Simulates a video feed by reading from video file at exactly 30 FPS."""
    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path
        self.cap = None
        self.frame_time = 1.0 / 30.0  # 30 FPS
        self.last_frame_time = 0

    def initialize(self) -> bool:
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            print(f"Failed to open video: {self.video_path}")
            return False
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Spoof video feed initialized: {fps:.1f} FPS source, {frame_count} frames")
        print("Will simulate 30 FPS video feed")
        return True

    def read_frame(self):
        current_time = time.time()
        # Wait to maintain 30 FPS
        if self.last_frame_time > 0:
            elapsed = current_time - self.last_frame_time
            if elapsed < self.frame_time:
                time.sleep(self.frame_time - elapsed)
        self.last_frame_time = time.time()
        if self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None