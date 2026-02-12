"""Unit tests for web server module."""

import unittest
from unittest.mock import Mock, patch, MagicMock, AsyncMock, mock_open
import asyncio
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from src.server import WebServer
from src.camera import CameraCapture


class TestWebServer(unittest.TestCase):
    """Unit tests for WebServer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.server = WebServer(host='127.0.0.1', port=8080)
    
    def test_init(self):
        """Test server initialization."""
        self.assertEqual(self.server.host, '127.0.0.1')
        self.assertEqual(self.server.port, 8080)
        self.assertIsNone(self.server.app)
        self.assertIsNone(self.server.camera)
    
    def test_init_default_params(self):
        """Test server initialization with default parameters."""
        server = WebServer()
        self.assertEqual(server.host, '0.0.0.0')
        self.assertEqual(server.port, 8080)
    
    def test_set_camera(self):
        """Test setting camera instance."""
        mock_camera = Mock(spec=CameraCapture)
        self.server.set_camera(mock_camera)
        
        self.assertEqual(self.server.camera, mock_camera)
    
    def test_create_app(self):
        """Test application creation."""
        app = self.server.create_app()
        
        self.assertIsNotNone(app)
        self.assertIsInstance(app, web.Application)
        self.assertEqual(self.server.app, app)
    
    def test_create_app_routes(self):
        """Test that routes are created."""
        app = self.server.create_app()
        
        # Verify routes exist
        routes = [route for route in app.router.routes()]
        route_paths = [route.resource.canonical for route in routes]
        
        self.assertIn('/', route_paths)
        self.assertIn('/stream', route_paths)
    
    def test_get_host(self):
        """Test getting server host."""
        self.assertEqual(self.server.get_host(), '127.0.0.1')
    
    def test_get_port(self):
        """Test getting server port."""
        self.assertEqual(self.server.get_port(), 8080)


class TestWebServerAsync(AioHTTPTestCase):
    """Async unit tests for WebServer class."""
    
    async def get_application(self):
        """Create application for testing."""
        self.web_server = WebServer(host='127.0.0.1', port=8080)
        return self.web_server.create_app()
    
    @unittest_run_loop
    async def test_handle_index_success(self):
        """Test successful index page handling."""
        html_content = '<html><body>Test</body></html>'
        
        with patch('builtins.open', mock_open(read_data=html_content)):
            resp = await self.client.request('GET', '/')
            
            self.assertEqual(resp.status, 200)
            text = await resp.text()
            self.assertEqual(text, html_content)
    
    @unittest_run_loop
    async def test_handle_index_not_found(self):
        """Test index page not found."""
        with patch('builtins.open', side_effect=FileNotFoundError()):
            resp = await self.client.request('GET', '/')
            
            self.assertEqual(resp.status, 404)
            text = await resp.text()
            self.assertEqual(text, 'Index page not found')
    
    @unittest_run_loop
    async def test_handle_stream_no_camera(self):
        """Test stream handling without camera."""
        resp = await self.client.request('GET', '/stream')
        
        self.assertEqual(resp.status, 503)
        text = await resp.text()
        self.assertEqual(text, 'Camera not available')
    
    @unittest_run_loop
    async def test_handle_stream_camera_not_opened(self):
        """Test stream handling with closed camera."""
        mock_camera = Mock(spec=CameraCapture)
        mock_camera.is_opened.return_value = False
        
        self.web_server.set_camera(mock_camera)
        
        resp = await self.client.request('GET', '/stream')
        
        self.assertEqual(resp.status, 503)
    
    @unittest_run_loop
    async def test_handle_stream_success(self):
        """Test successful stream handling."""
        mock_camera = Mock(spec=CameraCapture)
        mock_camera.is_opened.return_value = True
        
        # Simulate returning a frame then None to end stream
        jpeg_data = b'\xff\xd8\xff\xe0test_jpeg_data'
        call_count = [0]
        
        def get_frame_side_effect():
            call_count[0] += 1
            if call_count[0] <= 2:
                return jpeg_data
            return None
        
        mock_camera.get_jpeg_frame.side_effect = get_frame_side_effect
        
        self.web_server.set_camera(mock_camera)
        
        # Make request and read first chunk
        resp = await self.client.request('GET', '/stream')
        
        self.assertEqual(resp.status, 200)
        self.assertEqual(
            resp.headers['Content-Type'],
            'multipart/x-mixed-replace; boundary=frame'
        )
        
        # Read some data
        chunk = await resp.content.read(100)
        self.assertGreater(len(chunk), 0)


class TestWebServerMethods(unittest.TestCase):
    """Unit tests for WebServer methods without async."""
    
    def test_set_camera_none(self):
        """Test setting camera to None."""
        server = WebServer()
        server.set_camera(None)
        
        self.assertIsNone(server.camera)
    
    @patch('src.server.web.run_app')
    def test_run(self, mock_run_app):
        """Test run method."""
        server = WebServer(host='127.0.0.1', port=9090)
        server.run()
        
        mock_run_app.assert_called_once()
        # Verify app was created
        self.assertIsNotNone(server.app)
    
    @patch('src.server.web.run_app')
    def test_run_with_existing_app(self, mock_run_app):
        """Test run method with pre-existing app."""
        server = WebServer()
        server.create_app()
        app = server.app
        
        server.run()
        
        # App should not be recreated
        self.assertEqual(server.app, app)
        mock_run_app.assert_called_once()


if __name__ == '__main__':
    unittest.main()
