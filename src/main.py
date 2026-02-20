class Application:
    """Stub Application class for test compatibility."""
    def __init__(self, camera_id=0, host='0.0.0.0', port=8080):
        self.camera_id = camera_id
        self.host = host
        self.port = port
        self.camera = None
        self.detector = None  # Legacy, not used with delegate pattern
        self.server = None
        import logging
        self.logger = logging.getLogger(__name__)

    def initialize(self):
        from src.camera_feed import CameraFeed
        self.camera = CameraFeed(camera_id=self.camera_id)
        result = self.camera.initialize()
        if not result:
            self.logger.error("Failed to initialize camera feed")
            return False
        from src.server import WebServer
        self.server = WebServer(host=self.host, port=self.port)
        self.server.set_video_feed(self.camera)
        self.app = self.server.create_app()
        return True

    def run(self):
        try:
            if not self.camera:
                raise RuntimeError("Camera not initialized. Call initialize() before run().")   
            if not self.server:
                raise RuntimeError("Server not initialized. Call initialize() before run().")
            if not self.app:
                raise RuntimeError("Application not created. Call create_app() before run().")
            self.server.run()
            return 0
        except KeyboardInterrupt:
            if self.camera:
                self.camera.release()
            return 0
        except Exception:
            if self.camera:
                self.camera.release()
            return 1
        finally:
            if self.camera:
                self.camera.release()

    def cleanup(self):
        if self.camera:
            self.camera.release()
        self.camera = None
        self.detector = None  # For legacy compatibility

"""Main entry point for vision location detector application.

Initializes video feed, detector postprocessor, and web server, then starts the application.
"""


import sys
import logging
from src.camera_feed import CameraFeed
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

    # create video feed (camera or spoof) based on settings
    video_feed = CameraFeed(
        camera_id=settings.camera_id,
        width=settings.camera_width,
        height=settings.camera_height,
    )

    # Initialize video feed and detector
    detector = None
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

        video_feed.detection_delegate = detector  # Set the detector as the delegate for processing frames

    if not detector:
        logger.info("YOLO disabled by configuration (ENABLE_YOLO=false)")

    # Initialize and start application
    logger.info(f"Initializing application on {settings.server_host}:{settings.server_port}...")
    application = Application(camera_id=settings.camera_id, host=settings.server_host, port=settings.server_port)
    if not application.initialize():
        logger.error("Failed to initialize application")
        return 1
    logger.info("Starting application...")
    return application.run()
if __name__ == '__main__':
    sys.exit(main())
