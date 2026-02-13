"""Main entry point for vision location detector application.

This module initializes the camera and web server, and starts the application.
"""

import sys
import logging
from src.camera import CameraCapture
from src.server import WebServer
from src.detector import PedestrianDetector
from src.settings import settings


class Application:
    """Main application class for vision location detector."""
    
    def __init__(self):
        """Initialize the application using settings from environment."""
        self.camera = None
        self.detector = None
        self.server = None
        self._setup_logging()
        self.logger.info(f"Application settings: {settings}")
    
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_camera(self) -> bool:
        """Initialize the camera.
        
        Returns:
            True if initialization successful, False otherwise
        """
        self.logger.info(f"Initializing camera {settings.camera_id}...")
        self.camera = CameraCapture(
            camera_id=settings.camera_id,
            width=settings.camera_width,
            height=settings.camera_height
        )
        
        if not self.camera.initialize():
            self.logger.error("Failed to initialize camera")
            return False
        
        self.logger.info("Camera initialized successfully")
        return True
    
    def initialize_detector(self) -> bool:
        """Initialize the pedestrian detector.
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not settings.enable_yolo:
            self.logger.info("YOLO disabled by configuration (ENABLE_YOLO=false)")
            return True
        
        self.logger.info("Initializing YOLO pedestrian detector...")
        self.detector = PedestrianDetector(
            model_path=settings.yolo_model_path,
            confidence_threshold=settings.confidence_threshold,
            use_tensorrt=settings.use_tensorrt,
            device=settings.device
        )
        
        if not self.detector.initialize():
            self.logger.error("Failed to initialize detector")
            return False
        
        self.logger.info("Detector initialized successfully")
        return True
    
    def initialize_server(self) -> None:
        """Initialize the web server."""
        self.logger.info(f"Initializing web server on {settings.server_host}:{settings.server_port}...")
        self.server = WebServer(host=settings.server_host, port=settings.server_port)
        
        # Type narrowing: ensure camera was initialized
        if self.camera is not None:
            self.server.set_camera(self.camera)
        
        # Set detector if available
        if self.detector is not None:
            self.server.set_detector(self.detector)
        
        self.logger.info("Web server initialized")
    
    def run(self) -> int:
        """Run the application.
        
        Returns:
            Exit code (0 for success, 1 for failure)
        """
        try:
            # Initialize camera
            if not self.initialize_camera():
                return 1
            
            # Initialize detector with Jetson GPU acceleration (optional)
            # If this fails, we still run with camera-only mode
            if not self.initialize_detector():
                self.logger.warning("Failed to initialize detector - running in camera-only mode")
            
            # Initialize server
            self.initialize_server()
            
            # Start server (type narrowing check)
            if self.server is None:
                self.logger.error("Server not initialized")
                return 1
            
            # Start server
            self.logger.info("Starting application...")
            self.logger.info(f"Access the application at http://{settings.server_host}:{settings.server_port}")
            self.server.run()
            
            return 0
            
        except KeyboardInterrupt:
            self.logger.info("Application stopped by user")
            return 0
        except Exception as e:
            self.logger.error(f"Application error: {e}", exc_info=True)
            return 1
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.camera is not None:
            self.logger.info("Releasing camera...")
            self.camera.release()
        
        if self.detector is not None:
            self.logger.info("Releasing detector...")
            self.detector.release()


def main() -> int:
    """Main function.
    
    Returns:
        Exit code
    """
    # Create and run application (settings loaded from environment/.env)
    app = Application()
    return app.run()


if __name__ == '__main__':
    sys.exit(main())
