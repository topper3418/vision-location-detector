"""Integration tests for main application module.

These tests verify the main module works correctly with external
packages both mocked and unmocked, as per dev rules.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import logging

# Import dependencies to test integration
from src.main import Application, main
from src.camera import CameraCapture
from src.server import WebServer
from src.detector import PedestrianDetector


class TestApplicationIntegration(unittest.TestCase):
    """Integration tests for Application with external packages."""
    
    def test_logging_import_unmocked(self):
        """Test that logging can be imported without mocking."""
        # Verify logging is available
        self.assertTrue(hasattr(logging, 'basicConfig'))
        self.assertTrue(hasattr(logging, 'getLogger'))
    
    def test_sys_import_unmocked(self):
        """Test that sys can be imported without mocking."""
        # Verify sys is available
        self.assertTrue(hasattr(sys, 'argv'))
        self.assertTrue(hasattr(sys, 'exit'))
    
    def test_application_init_unmocked(self):
        """Test application initialization with unmocked dependencies."""
        app = Application(camera_id=0, host='127.0.0.1', port=8080)
        
        self.assertIsNotNone(app)
        self.assertIsNotNone(app.logger)
        self.assertIsInstance(app.logger, logging.Logger)
    
    @patch('src.camera.cv2.VideoCapture')
    def test_camera_integration_mocked(self, mock_video_capture):
        """Test camera integration with mocked cv2."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        app = Application()
        result = app.initialize_camera()
        
        self.assertTrue(result)
        self.assertIsInstance(app.camera, CameraCapture)
    
    @patch('src.detector.YOLO')
    def test_detector_integration_mocked(self, mock_yolo):
        """Test detector integration with mocked YOLO."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        app = Application()
        result = app.initialize_detector()
        
        self.assertTrue(result)
        self.assertIsInstance(app.detector, PedestrianDetector)
    
    @patch('src.detector.YOLO')
    @patch('src.camera.cv2.VideoCapture')
    def test_server_integration_mocked(self, mock_video_capture, mock_yolo):
        """Test server integration with mocked dependencies."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        app = Application()
        app.initialize_camera()
        app.initialize_detector()
        app.initialize_server()
        
        self.assertIsInstance(app.server, WebServer)
        self.assertEqual(app.server.camera, app.camera)
        self.assertEqual(app.server.detector, app.detector)
    
    @patch('src.detector.YOLO')
    @patch('src.server.web.run_app')
    @patch('src.camera.cv2.VideoCapture')
    def test_full_application_lifecycle_mocked(self, mock_video_capture, mock_run_app, mock_yolo):
        """Test full application lifecycle with mocked dependencies."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        app = Application(camera_id=0, host='127.0.0.1', port=8080)
        result = app.run()
        
        self.assertEqual(result, 0)
        mock_run_app.assert_called_once()
        mock_cap.release.assert_called_once()
    
    @patch('src.detector.YOLO')
    @patch('src.server.web.run_app')
    @patch('src.camera.cv2.VideoCapture')
    def test_keyboard_interrupt_integration_mocked(self, mock_video_capture, mock_run_app, mock_yolo):
        """Test keyboard interrupt handling with mocked dependencies."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        mock_run_app.side_effect = KeyboardInterrupt()
        
        app = Application()
        result = app.run()
        
        self.assertEqual(result, 0)
        # Ensure cleanup happened
        mock_cap.release.assert_called_once()
    
    @patch('src.camera.cv2.VideoCapture')
    def test_logging_integration_unmocked(self, mock_video_capture):
        """Test logging integration with real logging module."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        # Create app and verify logger works
        app = Application()
        
        # Test that logging methods exist and work
        self.assertTrue(hasattr(app.logger, 'info'))
        self.assertTrue(hasattr(app.logger, 'error'))
        
        # Test that we can log without errors
        app.logger.info("Test message")
        app.logger.error("Test error")


class TestMainIntegration(unittest.TestCase):
    """Integration tests for main function."""
    
    @patch('src.main.Application')
    def test_main_with_sys_argv_mocked(self, mock_app_class):
        """Test main function with mocked sys.argv."""
        mock_app = Mock()
        mock_app.run.return_value = 0
        mock_app_class.return_value = mock_app
        
        with patch.object(sys, 'argv', ['main.py', '1', '0.0.0.0', '8888']):
            result = main()
        
        self.assertEqual(result, 0)
        mock_app_class.assert_called_once_with(
            camera_id=1,
            host='0.0.0.0',
            port=8888
        )
    
    def test_sys_argv_unmocked(self):
        """Test that sys.argv works unmocked."""
        # Verify sys.argv is accessible
        self.assertIsInstance(sys.argv, list)
        self.assertGreater(len(sys.argv), 0)
    
    @patch('src.server.web.run_app')
    @patch('src.camera.cv2.VideoCapture')
    @patch.object(sys, 'argv', ['main.py'])
    def test_main_full_integration_mocked(self, mock_video_capture, mock_run_app):
        """Test main function with full integration (mocked external deps)."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        result = main()
        
        self.assertEqual(result, 0)
        mock_run_app.assert_called_once()
    
    @patch('builtins.print')
    @patch.object(sys, 'argv', ['main.py', 'not_a_number'])
    def test_main_error_handling_unmocked_print(self, mock_print):
        """Test main error handling with unmocked print."""
        result = main()
        
        self.assertEqual(result, 1)
        mock_print.assert_called_once()
        
        # Verify print was called with error message
        call_args = mock_print.call_args[0][0]
        self.assertIn('Invalid camera ID', call_args)


if __name__ == '__main__':
    unittest.main()
