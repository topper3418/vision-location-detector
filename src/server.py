"""Web server module for streaming camera feed.

This module handles HTTP requests and video streaming using aiohttp.
"""

import asyncio
from aiohttp import web
from typing import Optional, List
import os
import cv2
import json

from src.camera import CameraCapture
from src.camera_base import CameraBase
from src.detector import PedestrianDetector, DetectionResult
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
        self.camera: Optional[CameraBase] = None
        self.detector: Optional[PedestrianDetector] = None
        self.latest_detections: List[DetectionResult] = []
        self._latest_frame = None
        self._detection_lock = asyncio.Lock()
        self._detection_task = None

    async def _detection_loop(self):
        import numpy as np
        import cv2
        while True:
            if self._latest_frame is not None and self.detector is not None:
                frame = self._latest_frame
                self._latest_frame = None  # Drop all but the latest
                # Debug: print frame shape and dtype
                print(f"[DEBUG] Frame shape: {frame.shape}, dtype: {frame.dtype}")
                # Convert BGR (OpenCV) to RGB for YOLO if needed
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                async with self._detection_lock:
                    detections = self.detector.detect(rgb_frame)
                    self.latest_detections = detections
                    self._latest_processed_frame = self.detector.draw_detections(frame, detections)
            await asyncio.sleep(0.001)
    
    def set_camera(self, camera: CameraBase) -> None:
        """Set the camera capture instance (supports any CameraBase subclass)."""
        self.camera = camera
    
    def set_detector(self, detector: PedestrianDetector) -> None:
        """Set the pedestrian detector instance.
        
        Args:
            detector: PedestrianDetector instance to use for detection
        """
        self.detector = detector
    
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
        
        try:
            with open(html_path, 'r') as f:
                html_content = f.read()
            return web.Response(text=html_content, content_type='text/html')
        except FileNotFoundError:
            return web.Response(text='Index page not found', status=404)
    
    async def handle_stream(self, request: web.Request) -> web.StreamResponse:
        """Handle video stream request.
        
        Args:
            request: HTTP request object
            
        Returns:
            Streaming HTTP response with MJPEG video
        """
        if self.camera is None or not self.camera.is_opened():
            return web.Response(text='Camera not available', status=503)
        
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
        # Start detection loop if not already running
        if self._detection_task is None:
            self._detection_task = asyncio.create_task(self._detection_loop())

        try:
            while True:
                # Always grab the latest frame
                success, frame = self.camera.read_frame()
                if not success or frame is None:
                    print("CAMERA READ FAILED: No frame captured")
                    await asyncio.sleep(0.01)
                    continue
                self._latest_frame = frame

                # Wait for detection to finish and get the latest processed frame
                processed_frame = getattr(self, '_latest_processed_frame', frame)

                # Encode processed frame as JPEG
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), settings.jpeg_quality]
                success, buffer = cv2.imencode('.jpg', processed_frame, encode_params)
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
        detector_enabled = self.detector is not None
        
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
        if self.camera is None:
            fps = 0.0
        else:
            fps = self.camera.get_fps()
        
        fps_data = {
            'fps': round(fps, 1),
            'timestamp': asyncio.get_event_loop().time()
        }
        
        return web.json_response(fps_data)
    
    async def handle_settings(self, request: web.Request) -> web.Response:
        """Handle settings request.
        
        Args:
            request: HTTP request object
            
        Returns:
            JSON response with application settings
        """
        settings_data = {
            'device': settings.device,
            'enable_yolo': settings.enable_yolo,
            'use_tensorrt': settings.use_tensorrt,
            'yolo_model_path': settings.yolo_model_path,
            'confidence_threshold': settings.confidence_threshold,
            'camera_id': settings.camera_id,
            'camera_width': settings.camera_width,
            'camera_height': settings.camera_height,
            'server_host': settings.server_host,
            'server_port': settings.server_port,
            'jpeg_quality': settings.jpeg_quality
        }
        
        return web.json_response(settings_data)
    
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
        
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
    
    def run(self) -> None:
        """Run the web server (blocking).
        
        This is a convenience method that creates the app and runs it.
        """
        if self.app is None:
            self.create_app()
        
        web.run_app(self.app, host=self.host, port=self.port)
