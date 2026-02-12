"""Main entry point for vision location detector application.

This module initializes the camera and web server, and starts the application.
"""

import sys
import logging
from src.camera import CameraCapture
from src.server import WebServer


class Application:
    """Main application class for vision location detector."""
    
    def __init__(self, camera_id: int = 0, host: str = '0.0.0.0', port: int = 8080):
        """Initialize the application.
        
        Args:
            camera_id: Camera device ID
            host: Server host address
            port: Server port number
        """
        self.camera_id = camera_id
        self.host = host
        self.port = port
        self.camera = None
        self.server = None
        self._setup_logging()
    
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
        self.logger.info(f"Initializing camera {self.camera_id}...")
        self.camera = CameraCapture(camera_id=self.camera_id)
        
        if not self.camera.initialize():
            self.logger.error("Failed to initialize camera")
            return False
        
        self.logger.info("Camera initialized successfully")
        return True
    
    def initialize_server(self) -> None:
        """Initialize the web server."""
        self.logger.info(f"Initializing web server on {self.host}:{self.port}...")
        self.server = WebServer(host=self.host, port=self.port)
        self.server.set_camera(self.camera)
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
            
            # Initialize server
            self.initialize_server()
            
            # Start server
            self.logger.info("Starting application...")
            self.logger.info(f"Access the application at http://{self.host}:{self.port}")
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


def main() -> int:
    """Main function.
    
    Returns:
        Exit code
    """
    # Parse command line arguments (basic)
    camera_id = 0
    host = '0.0.0.0'
    port = 8080
    
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera ID: {sys.argv[1]}")
            return 1
    
    if len(sys.argv) > 2:
        host = sys.argv[2]
    
    if len(sys.argv) > 3:
        try:
            port = int(sys.argv[3])
        except ValueError:
            print(f"Invalid port: {sys.argv[3]}")
            return 1
    
    # Create and run application
    app = Application(camera_id=camera_id, host=host, port=port)
    return app.run()


if __name__ == '__main__':
    sys.exit(main())
