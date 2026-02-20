"""Script to run pedestrian detection on a recorded video using PedestrianDetector."""
import os
import sys
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.util.get_video_path import get_video_path
from src.detection_services.pedestrian_detector import PedestrianDetector
from src.video_feeds.spoof_video_feed import SpoofVideoFeed
from src.interfaces.video_feed_base import VideoFeedBase
from src.server_builder import ServerBuilder


def main():
    video_path = get_video_path()
    if not video_path:
        return
    print(f"Analyzing video: {video_path}")

    # Use spoof video feed to simulate 30 FPS video feed
    video_feed: VideoFeedBase = SpoofVideoFeed(video_path)
    builder = ServerBuilder()
    builder.video_feed = video_feed
    builder.initialize()  # This will initialize the video feed and detector

    frame_count = 0
    detection_count = 0
    total_detections = 0
    max_detections_per_frame = 0
    start_time = time.time()

    print("Processing frames at exact 30 FPS simulation...")

    for detections in video_feed.get_data_stream():
        frame_count += 1
        if detections:
            detection_count += 1
            total_detections += len(detections)
            max_detections_per_frame = max(max_detections_per_frame, len(detections))
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            actual_fps = video_feed.fps
            print(f"Frame {frame_count}: {len(detections) if detections else 0} detection(s) | Actual FPS: {actual_fps:.1f}")

    video_feed.release()

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
