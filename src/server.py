"""Web server module for streaming camera feed.

This module handles HTTP requests and video streaming using aiohttp.
"""

import asyncio
from aiohttp import web
from typing import Optional, List
import os
import cv2

from src.interfaces.video_feed_base import VideoFeedBase
from src.interfaces.detection_result import DetectionResult
from src.settings import settings


@web.middleware
async def cors_middleware(request: web.Request, handler):
    """Add CORS headers to all responses."""
    response = await handler(request)
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


class WebServer:
    """Handles HTTP server and video streaming."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        """Initialize web server.
        
        Args:
            host: Server host address
            port: Server port number
        """
        self.host = host
        self.port = port
        self.app: Optional[web.Application] = None
        self.video_feed: Optional[VideoFeedBase] = None
        self.latest_detections: List[DetectionResult] = []
        self.processor = None  # DetectionProcessor

    # Detection loop is now handled by the video feed's postprocessing pipeline
    
    def set_video_feed(self, video_feed: VideoFeedBase) -> None:
        """Set the video feed instance (supports any VideoFeedBase subclass)."""
        self.video_feed = video_feed
    
    def create_app(self) -> web.Application:
        """Create and configure the aiohttp application.
        
        Returns:
            Configured web.Application instance
        """
        app = web.Application(middlewares=[cors_middleware])
        
        # Add routes
        app.router.add_get('/', self.handle_index)
        app.router.add_get('/stream', self.handle_stream)
        app.router.add_get('/detections', self.handle_detections)
        app.router.add_get('/fps', self.handle_fps)
        app.router.add_get('/settings', self.handle_settings)
        
        self.app = app
        return app
    
    async def handle_index(self, request: web.Request) -> web.Response:
        """Handle index page request.
        
        Args:
            request: HTTP request object
            
        Returns:
            HTTP response with HTML content
        """
        html_path = os.path.join(
            os.path.dirname(__file__), 
            'static', 
            'index.html'
        )
        
        with open(html_path, 'r') as f:
            html_content = f.read()
        return web.Response(text=html_content, content_type='text/html')
    
    async def handle_stream(self, request: web.Request) -> web.StreamResponse:
        """Handle video stream request.
        
        Args:
            request: HTTP request object
            
        Returns:
            Streaming HTTP response with MJPEG video
        """
        if self.video_feed is None or not self.video_feed.is_opened():
            return web.Response(text='Video feed not available', status=503)
        
        # Create multipart response for MJPEG stream
        response = web.StreamResponse(
            status=200,
            reason='OK',
            headers={
                'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
        )
        
        await response.prepare(request)
        # No detection loop needed; handled by video feed postprocessors

        try:
            # Use the video feed's full stream (yields (data, processed) tuples)
            for data, processed in self.video_feed.get_full_stream():
                # If data is a list of detections, update latest_detections
                if isinstance(data, list) and data and hasattr(data[0], 'to_dict'):
                    self.latest_detections = data
                # Encode processed frame as JPEG
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality]
                success, buffer = cv2.imencode('.jpg', processed, encode_params)
                if not success:
                    print("JPEG ENCODE FAILED: Unable to encode frame")
                    await asyncio.sleep(0.01)
                    continue
                jpeg_data = buffer.tobytes()
                await response.write(
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + 
                    jpeg_data + 
                    b'\r\n'
                )
                await asyncio.sleep(0)
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            await response.write_eof()
        return response
    
    async def handle_detections(self, request: web.Request) -> web.Response:
        """Handle detections JSON request.
        
        Args:
            request: HTTP request object
            
        Returns:
            JSON response with current detection results
        """
        # Check if detector is enabled
        detector_enabled = bool(self.video_feed and self.video_feed.detection_delegate)
        detections_data = {
            'enabled': detector_enabled,
            'count': len(self.latest_detections),
            'detections': [det.to_dict() for det in self.latest_detections],
            'timestamp': asyncio.get_event_loop().time()
        }
        return web.json_response(detections_data)
    
    async def handle_fps(self, request: web.Request) -> web.Response:
        """Handle FPS request.
        
        Args:
            request: HTTP request object
            
        Returns:
            JSON response with current FPS
        """
        if self.video_feed is None or not self.video_feed.is_opened() or not self.video_feed.measure_fps:
            fps = 0.0
        else:
            fps = self.video_feed.fps
        
        fps_data = {
            'fps': round(fps, 1),
            'timestamp': asyncio.get_event_loop().time()
        }
        
        return web.json_response(fps_data)
    
    async def handle_settings(self, request: web.Request) -> web.Response:
        """
        Handle settings request.
        Args:
            request: HTTP request object
        Returns:
            JSON response with application settings
        """
        return web.json_response(settings.to_dict())
    
    def get_host(self) -> str:
        """Get server host.
        
        Returns:
            Server host address
        """
        return self.host
    
    def get_port(self) -> int:
        """Get server port.
        
        Returns:
            Server port number
        """
        return self.port
    
    async def start(self) -> None:
        """Start the web server."""
        if self.app is None:
            self.create_app()
        
        if self.app is None:
            raise RuntimeError("Application not created. Cannot start server.")
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
    
    def run(self) -> None:
        """Run the web server (blocking).
        
        Raises:
            RuntimeError: If application is not created before running
        """
        if self.app is None:
            raise RuntimeError("Application not created. Call create_app() before run().")
        
        web.run_app(self.app, host=self.host, port=self.port)
