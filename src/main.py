
"""Main entry point for vision location detector application.

Initializes video feed, detector postprocessor, and web server, then starts the application.
"""


import sys
import logging
from src.camera import CameraFeed
from src.server import WebServer
from src.detector import PedestrianDetector
from src.settings import settings

def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Application settings: {settings}")

    # Initialize video feed
    video_feed = CameraFeed(
        camera_id=settings.camera_id,
        width=settings.camera_width,
        height=settings.camera_height
    )
    logger.info(f"Initializing camera {settings.camera_id}...")
    if not video_feed.initialize():
        logger.error("Failed to initialize camera")
        video_feed.release()
        return 1
    logger.info("Camera initialized successfully")

    # Initialize detector and add as postprocessor if enabled
    if settings.enable_yolo:
        logger.info("Initializing YOLO pedestrian detector...")
        detector = PedestrianDetector(
            model_path=settings.yolo_model_path,
            confidence_threshold=settings.confidence_threshold,
            use_tensorrt=settings.use_tensorrt,
            device=settings.device
        )
        if not detector.initialize():
            raise RuntimeError("Failed to initialize YOLO detector")
        else:
            logger.info("Detector initialized successfully")
            video_feed.add_postprocessor(detector.detect)
    else:
        logger.info("YOLO disabled by configuration (ENABLE_YOLO=false)")

    # Initialize and start server
    logger.info(f"Initializing web server on {settings.server_host}:{settings.server_port}...")
    server = WebServer(host=settings.server_host, port=settings.server_port)
    server.set_video_feed(video_feed)
    logger.info("Web server initialized")
    logger.info("Starting application...")
    logger.info(f"Access the application at http://{settings.server_host}:{settings.server_port}")
    try:
        server.run()
        return 0
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1
    finally:
        logger.info("Releasing camera...")
        video_feed.release()

if __name__ == '__main__':
    sys.exit(main())
