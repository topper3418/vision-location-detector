"""Script to run pedestrian detection on a recorded video using PedestrianDetector."""
import cv2
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.detector import PedestrianDetector
from src.spoof_camera import SpoofCamera
from src.camera_base import CameraBase


def main():
    captures_dir = os.path.join(os.path.dirname(__file__), '..', 'captures')
    video_path = None
    if len(sys.argv) >= 2:
        # If argument is given, use it (search in captures/ if not absolute)
        arg_path = sys.argv[1]
        if not os.path.isabs(arg_path):
            candidate = os.path.join(captures_dir, arg_path)
            if os.path.exists(candidate):
                video_path = candidate
            elif os.path.exists(arg_path):
                video_path = arg_path
            else:
                print(f"Video file not found: {arg_path}")
                return
        else:
            if os.path.exists(arg_path):
                video_path = arg_path
            else:
                print(f"Video file not found: {arg_path}")
                return
    else:
        # No argument: use latest file in captures/
        if not os.path.exists(captures_dir):
            print("No captures/ directory found. Please record a video first.")
            return
        files = [f for f in os.listdir(captures_dir) if f.lower().endswith(('.avi', '.mp4', '.mov'))]
        if not files:
            print("No video files found in captures/. Please record a video first.")
            return
        files.sort(key=lambda f: os.path.getmtime(os.path.join(captures_dir, f)), reverse=True)
        video_path = os.path.join(captures_dir, files[0])
        print(f"No video argument given. Using latest capture: {files[0]}")

    print(f"Analyzing video: {video_path}")
    
    # Use spoof camera to simulate 30 FPS camera feed

    camera: CameraBase = SpoofCamera(video_path)
    if not camera.initialize():
        print(f"Failed to initialize spoof camera for video: {video_path}")
        return

    detector = PedestrianDetector()
    if not detector.initialize():
        print("Failed to initialize detector.")
        return

    # Add detection as a postprocessor to the camera
    def detection_postprocessor(frame):
        import cv2
        print(f"[DEBUG] Frame shape: {frame.shape}, dtype: {frame.dtype}")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detector.detect(rgb_frame)
        annotated = detector.draw_detections(frame, detections)
        return (annotated, detections)
    camera.add_postprocessor(detection_postprocessor)

    frame_count = 0
    detection_count = 0
    total_detections = 0
    max_detections_per_frame = 0
    start_time = time.time()

    print("Processing frames at exact 30 FPS simulation...")

    for annotated_frame, detections in camera.get_processed_stream():
        frame_count += 1
        if detections:
            detection_count += 1
            total_detections += len(detections)
            max_detections_per_frame = max(max_detections_per_frame, len(detections))
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            elapsed = time.time() - start_time
            actual_fps = frame_count / elapsed if elapsed > 0 else 0
            print(f"Frame {frame_count}: {len(detections)} detection(s) | Actual FPS: {actual_fps:.1f}")

    camera.release()

    # Calculate overall FPS
    total_time = time.time() - start_time
    overall_fps = frame_count / total_time if total_time > 0 else 0

    # Print results
    print("\n" + "="*50)
    print("DETECTION ANALYSIS RESULTS")
    print("="*50)
    print(f"Total frames processed: {frame_count}")
    print(f"Frames with detections: {detection_count}")
    print(f"Total detections found: {total_detections}")
    print(f"Max detections per frame: {max_detections_per_frame}")
    print(f"Overall FPS: {overall_fps:.2f}")

    if detection_count > 0:
        avg_detections = total_detections / detection_count
        detection_rate = (detection_count / frame_count) * 100
        print("✅ DETECTION WORKING - Pedestrians detected!")
    else:
        print("❌ NO DETECTIONS FOUND - Check detector configuration")
        print("   - Verify ENABLE_YOLO=true in .env")
        print("   - Check YOLO model path")
        print("   - Ensure video contains visible pedestrians")

    print("="*50)

if __name__ == "__main__":
    main()
