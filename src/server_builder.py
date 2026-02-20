from src.video_feeds.camera_feed import CameraFeed
from src.detection_services.pedestrian_detector import PedestrianDetector
from src.server import WebServer
from src.settings import settings
import logging


class ServerBuilder:
    def __init__(self, 
                 camera_id=None, camera_width=None, camera_height=None,
                 enable_yolo=None, yolo_model_path=None, confidence_threshold=None,
                 use_tensorrt=None, device=None,
                 server_host=None, server_port=None):
        self.logger = logging.getLogger(__name__)
        # Use provided params or fallback to settings
        self.camera_id = camera_id if camera_id is not None else settings.camera_id
        self.camera_width = camera_width if camera_width is not None else settings.camera_width
        self.camera_height = camera_height if camera_height is not None else settings.camera_height
        self.enable_yolo = enable_yolo if enable_yolo is not None else settings.enable_yolo
        self.yolo_model_path = yolo_model_path if yolo_model_path is not None else settings.yolo_model_path
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else settings.confidence_threshold
        self.use_tensorrt = use_tensorrt if use_tensorrt is not None else settings.use_tensorrt
        self.device = device if device is not None else settings.device
        self.server_host = server_host if server_host is not None else settings.server_host
        self.server_port = server_port if server_port is not None else settings.server_port

        self.video_feed = None
        self.detector = None
        self.server = None

    def initialize(self):
        """Initializes video feed and detector objects, unless already set."""
        # Video feed: if already set and initialized, skip
        if self.video_feed is not None:
            if hasattr(self.video_feed, 'initialized') and self.video_feed.initialized:
                self.logger.info("Using provided video feed (already initialized).")
            else:
                self.logger.info("Using provided video feed, initializing...")
                self.video_feed.measure_fps = True
                if not self.video_feed.initialize():
                    self.logger.error("Failed to initialize provided camera feed")
                    return False
        else:
            self.logger.info("Initializing new video feed...")
            self.video_feed = CameraFeed(
                camera_id=self.camera_id,
                width=self.camera_width,
                height=self.camera_height,
            )
            self.video_feed.measure_fps = True
            if not self.video_feed.initialize():
                self.logger.error("Failed to initialize camera feed")
                return False

        # Detector: if already set and initialized, skip
        if self.enable_yolo:
            if self.detector is not None:
                if hasattr(self.detector, 'initialized') and self.detector.initialized:
                    self.logger.info("Using provided detector (already initialized).")
                else:
                    self.logger.info("Using provided detector, initializing...")
                    if not self.detector.initialize():
                        self.logger.error("Failed to initialize provided detector")
                        self.video_feed.release()
                        return False
                    else:
                        self.logger.info("Detector initialized successfully")
                self.video_feed.set_detector_delegate(self.detector)
            else:
                self.logger.info("Initializing new YOLO pedestrian detector...")
                self.detector = PedestrianDetector(
                    model_path=self.yolo_model_path,
                    confidence_threshold=self.confidence_threshold,
                    use_tensorrt=self.use_tensorrt,
                    device=self.device
                )
                if not self.detector.initialize():
                    self.logger.error("Failed to initialize YOLO detector")
                    self.video_feed.release()
                    return False
                else:
                    self.logger.info("Detector initialized successfully")
                self.video_feed.set_detector_delegate(self.detector)
        else:
            self.logger.info("YOLO disabled by configuration (ENABLE_YOLO=false)")
        return True

    def build(self):
        """Builds and returns the web server and video feed objects, fully initialized."""
        if not self.video_feed:
            if not self.initialize():
                return None, None
        self.logger.info("Building web server...")
        self.server = WebServer(host=self.server_host, port=self.server_port)
        self.server.set_video_feed(self.video_feed)
        self.server.create_app()
        return self.server, self.video_feed