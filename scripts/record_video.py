"""Script to record a video from the camera using the CameraCapture class."""
import cv2
import time
import sys
import os
import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.camera import CameraCapture
import os

def main():

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'captures')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f"capture_{timestamp}.avi")

    camera = CameraCapture()
    if not camera.initialize():
        print("Failed to initialize camera.")
        return

    width = camera.width
    height = camera.height
    fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Recording video to {output_path}. Press Ctrl+C to stop.")
    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                print("Failed to read frame from camera.")
                break
            out.write(frame)
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        out.release()
        camera.release()
        print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main()
