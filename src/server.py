"""Web server module for streaming camera feed.

This module handles HTTP requests and video streaming using aiohttp.
"""

import asyncio
from aiohttp import web
from typing import Optional
import os

from src.camera import CameraCapture


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
        self.camera: Optional[CameraCapture] = None
    
    def set_camera(self, camera: CameraCapture) -> None:
        """Set the camera capture instance.
        
        Args:
            camera: CameraCapture instance to use for streaming
        """
        self.camera = camera
    
    def create_app(self) -> web.Application:
        """Create and configure the aiohttp application.
        
        Returns:
            Configured web.Application instance
        """
        app = web.Application()
        
        # Add routes
        app.router.add_get('/', self.handle_index)
        app.router.add_get('/stream', self.handle_stream)
        
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
        
        try:
            while True:
                # Get JPEG frame from camera
                jpeg_data = self.camera.get_jpeg_frame()
                
                if jpeg_data is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Send frame as multipart data
                await response.write(
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + 
                    jpeg_data + 
                    b'\r\n'
                )
                
                # Small delay to control frame rate
                await asyncio.sleep(0.033)  # ~30 FPS
                
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        finally:
            await response.write_eof()
        
        return response
    
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
