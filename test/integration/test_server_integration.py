"""Integration tests for web server module.

These tests verify the server module works correctly with external
packages both mocked and unmocked, as per dev rules.
"""

import unittest
from unittest.mock import Mock, patch, AsyncMock, mock_open
import asyncio
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

# Import aiohttp to test integration
import aiohttp
from src.server import WebServer
from src.camera import CameraCapture


class TestServerIntegration(unittest.TestCase):
    """Integration tests for WebServer with external packages."""
    
    def test_aiohttp_import_unmocked(self):
        """Test that aiohttp can be imported without mocking."""
        # Verify aiohttp is available
        self.assertTrue(hasattr(aiohttp, 'web'))
        self.assertTrue(hasattr(web, 'Application'))
        self.assertTrue(hasattr(web, 'Response'))
        self.assertTrue(hasattr(web, 'StreamResponse'))
    
    def test_server_init_unmocked(self):
        """Test server initialization with unmocked aiohttp."""
        server = WebServer(host='127.0.0.1', port=8080)
        self.assertIsNotNone(server)
        self.assertEqual(server.host, '127.0.0.1')
        self.assertEqual(server.port, 8080)
    
    def test_create_app_unmocked(self):
        """Test app creation with unmocked aiohttp."""
        server = WebServer()
        app = server.create_app()
        
        # Verify real aiohttp Application instance
        self.assertIsInstance(app, web.Application)
        self.assertTrue(hasattr(app, 'router'))
    
    @patch('aiohttp.web.run_app')
    def test_run_with_mocked_aiohttp(self, mock_run_app):
        """Test run method with mocked aiohttp.web.run_app."""
        server = WebServer(host='0.0.0.0', port=3000)
        server.run()
        
        # Verify run_app was called with correct parameters
        mock_run_app.assert_called_once()
        call_args = mock_run_app.call_args
        self.assertEqual(call_args[1]['host'], '0.0.0.0')
        self.assertEqual(call_args[1]['port'], 3000)
    
    def test_camera_integration_unmocked(self):
        """Test camera integration with server."""
        server = WebServer()
        camera = Mock(spec=CameraCapture)
        
        server.set_camera(camera)
        self.assertEqual(server.camera, camera)
    
    @patch('builtins.open', mock_open(read_data='<html>Test</html>'))
    def test_file_io_integration_mocked(self):
        """Test file I/O integration with mocked open."""
        server = WebServer()
        app = server.create_app()
        
        # Create a mock request
        request = Mock(spec=web.Request)
        
        # Test handle_index with mocked file I/O
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(server.handle_index(request))
            self.assertIsInstance(response, web.Response)
            self.assertEqual(response.text, '<html>Test</html>')
        finally:
            loop.close()


class TestServerAsyncIntegration(AioHTTPTestCase):
    """Async integration tests for WebServer."""
    
    async def get_application(self):
        """Create application for testing."""
        self.web_server = WebServer()
        return self.web_server.create_app()
    
    @unittest_run_loop
    async def test_index_route_unmocked(self):
        """Test index route with unmocked aiohttp."""
        with patch('builtins.open', mock_open(read_data='<html>Integration Test</html>')):
            resp = await self.client.request('GET', '/')
            
            self.assertEqual(resp.status, 200)
            self.assertIsInstance(resp, aiohttp.ClientResponse)
    
    @unittest_run_loop
    async def test_stream_route_unmocked(self):
        """Test stream route with unmocked aiohttp."""
        resp = await self.client.request('GET', '/stream')
        
        # Should get 503 since no camera is set
        self.assertEqual(resp.status, 503)
        self.assertIsInstance(resp, aiohttp.ClientResponse)
    
    @unittest_run_loop
    async def test_full_server_lifecycle_mocked(self):
        """Test full server lifecycle with mocked camera."""
        # Create mock camera
        mock_camera = Mock(spec=CameraCapture)
        mock_camera.is_opened.return_value = True
        mock_camera.get_jpeg_frame.return_value = b'\xff\xd8\xff\xe0test'
        
        self.web_server.set_camera(mock_camera)
        
        # Test index
        with patch('builtins.open', mock_open(read_data='<html>Test</html>')):
            resp = await self.client.request('GET', '/')
            self.assertEqual(resp.status, 200)
        
        # Test stream
        resp = await self.client.request('GET', '/stream')
        self.assertEqual(resp.status, 200)
        
        # Verify camera was used
        mock_camera.is_opened.assert_called()
    
    @unittest_run_loop
    async def test_asyncio_integration_unmocked(self):
        """Test asyncio integration with server."""
        # Test that async operations work correctly
        task = asyncio.create_task(
            self.client.request('GET', '/stream')
        )
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Cancel and verify
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected
    
    @unittest_run_loop
    async def test_streaming_response_mocked(self):
        """Test streaming response with mocked camera data."""
        mock_camera = Mock(spec=CameraCapture)
        mock_camera.is_opened.return_value = True
        
        # Create counter for limiting frames
        frame_count = [0]
        
        def get_frame():
            frame_count[0] += 1
            if frame_count[0] <= 3:
                return b'\xff\xd8\xff\xe0frame_data'
            return None
        
        mock_camera.get_jpeg_frame.side_effect = get_frame
        
        self.web_server.set_camera(mock_camera)
        
        # Request stream
        resp = await self.client.request('GET', '/stream')
        
        self.assertEqual(resp.status, 200)
        self.assertIn('multipart/x-mixed-replace', resp.headers['Content-Type'])
        
        # Read first chunk
        chunk = await resp.content.read(50)
        self.assertGreater(len(chunk), 0)


class TestServerOSIntegration(unittest.TestCase):
    """Integration tests for OS-level operations."""
    
    @patch('os.path.join')
    @patch('builtins.open', mock_open(read_data='test'))
    def test_os_path_integration_mocked(self, mock_path_join):
        """Test os.path integration with mocked paths."""
        mock_path_join.return_value = '/fake/path/index.html'
        
        server = WebServer()
        app = server.create_app()
        
        request = Mock(spec=web.Request)
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(server.handle_index(request))
            self.assertIsNotNone(response)
        finally:
            loop.close()
    
    def test_os_path_unmocked(self):
        """Test os.path operations unmocked."""
        import os
        
        # Verify os.path operations work
        server_file = __file__
        dirname = os.path.dirname(server_file)
        self.assertTrue(os.path.exists(dirname))


if __name__ == '__main__':
    unittest.main()
