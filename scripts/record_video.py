"""Script to record a video from the video feed using the VideoFeedCapture class."""
import cv2
import time
import sys
import os
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.camera_feed import CameraFeed
import os

def main():

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'captures')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f"capture_{timestamp}.avi")

    video_feed = CameraFeed()
    if not video_feed.initialize():
        print("Failed to initialize video feed.")
        return

    width = video_feed.width
    height = video_feed.height
    fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Recording video to {output_path}. Press Ctrl+C to stop.")
    try:
        while True:
            ret, frame = video_feed.read_frame()
            if not ret or frame is None:
                print("Failed to read frame from video feed.")
                break
            out.write(frame)
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        out.release()
        video_feed.release()
        print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()
