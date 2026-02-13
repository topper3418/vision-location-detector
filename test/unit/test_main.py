"""Unit tests for main application module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import logging

from src.main import Application, main


class TestApplication(unittest.TestCase):
    """Unit tests for Application class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.app = Application(camera_id=0, host='127.0.0.1', port=8080)
    
    def test_init(self):
        """Test application initialization."""
        self.assertEqual(self.app.camera_id, 0)
        self.assertEqual(self.app.host, '127.0.0.1')
        self.assertEqual(self.app.port, 8080)
        self.assertIsNone(self.app.camera)
        self.assertIsNone(self.app.detector)
        self.assertIsNone(self.app.server)
    
    def test_init_custom_params(self):
        """Test application initialization with custom parameters."""
        app = Application(camera_id=1, host='0.0.0.0', port=9090)
        self.assertEqual(app.camera_id, 1)
        self.assertEqual(app.host, '0.0.0.0')
        self.assertEqual(app.port, 9090)
    
    def test_setup_logging(self):
        """Test logging setup."""
        self.assertIsNotNone(self.app.logger)
        self.assertIsInstance(self.app.logger, logging.Logger)
    
    @patch('src.main.CameraCapture')
    def test_initialize_camera_success(self, mock_camera_class):
        """Test successful camera initialization."""
        mock_camera = Mock()
        mock_camera.initialize.return_value = True
        mock_camera_class.return_value = mock_camera
        
        result = self.app.initialize_camera()
        
        self.assertTrue(result)
        mock_camera_class.assert_called_once_with(camera_id=0)
        mock_camera.initialize.assert_called_once()
        self.assertEqual(self.app.camera, mock_camera)
    
    @patch('src.main.CameraCapture')
    def test_initialize_camera_failure(self, mock_camera_class):
        """Test failed camera initialization."""
        mock_camera = Mock()
        mock_camera.initialize.return_value = False
        mock_camera_class.return_value = mock_camera
        
        result = self.app.initialize_camera()
        
        self.assertFalse(result)
    
    @patch('src.main.PedestrianDetector')
    def test_initialize_detector_success(self, mock_detector_class):
        """Test successful detector initialization."""
        mock_detector = Mock()
        mock_detector.initialize.return_value = True
        mock_detector_class.return_value = mock_detector
        
        result = self.app.initialize_detector()
        
        self.assertTrue(result)
        mock_detector_class.assert_called_once_with()
        mock_detector.initialize.assert_called_once()
        self.assertEqual(self.app.detector, mock_detector)
    
    @patch('src.main.PedestrianDetector')
    def test_initialize_detector_failure(self, mock_detector_class):
        """Test failed detector initialization."""
        mock_detector = Mock()
        mock_detector.initialize.return_value = False
        mock_detector_class.return_value = mock_detector
        
        result = self.app.initialize_detector()
        
        self.assertFalse(result)
    
    @patch('src.main.WebServer')
    def test_initialize_server(self, mock_server_class):
        """Test server initialization."""
        mock_server = Mock()
        mock_server_class.return_value = mock_server
        
        mock_camera = Mock()
        self.app.camera = mock_camera
        
        mock_detector = Mock()
        self.app.detector = mock_detector
        
        self.app.initialize_server()
        
        mock_server_class.assert_called_once_with(host='127.0.0.1', port=8080)
        mock_server.set_camera.assert_called_once_with(mock_camera)
        mock_server.set_detector.assert_called_once_with(mock_detector)
        self.assertEqual(self.app.server, mock_server)
    
    @patch('src.main.PedestrianDetector')
    @patch('src.main.WebServer')
    @patch('src.main.CameraCapture')
    def test_run_success(self, mock_camera_class, mock_server_class, mock_detector_class):
        """Test successful application run."""
        # Setup mocks
        mock_camera = Mock()
        mock_camera.initialize.return_value = True
        mock_camera_class.return_value = mock_camera
        
        mock_detector = Mock()
        mock_detector.initialize.return_value = True
        mock_detector_class.return_value = mock_detector
        
        mock_server = Mock()
        mock_server_class.return_value = mock_server
        
        # Run application
        result = self.app.run()
        
        self.assertEqual(result, 0)
        mock_camera.initialize.assert_called_once()
        mock_detector.initialize.assert_called_once()
        mock_server.run.assert_called_once()
        mock_camera.release.assert_called_once()
        mock_detector.release.assert_called_once()
    
    @patch('src.main.CameraCapture')
    def test_run_camera_init_failure(self, mock_camera_class):
        """Test application run with camera initialization failure."""
        mock_camera = Mock()
        mock_camera.initialize.return_value = False
        mock_camera_class.return_value = mock_camera
        
        result = self.app.run()
        
        self.assertEqual(result, 1)
    
    @patch('src.main.PedestrianDetector')
    @patch('src.main.CameraCapture')
    def test_run_detector_init_failure(self, mock_camera_class, mock_detector_class):
        """Test application run with detector initialization failure."""
        mock_camera = Mock()
        mock_camera.initialize.return_value = True
        mock_camera_class.return_value = mock_camera
        
        mock_detector = Mock()
        mock_detector.initialize.return_value = False
        mock_detector_class.return_value = mock_detector
        
        result = self.app.run()
        
        self.assertEqual(result, 1)
    
    @patch('src.main.PedestrianDetector')
    @patch('src.main.WebServer')
    @patch('src.main.CameraCapture')
    def test_run_keyboard_interrupt(self, mock_camera_class, mock_server_class, mock_detector_class):
        """Test application run with keyboard interrupt."""
        mock_camera = Mock()
        mock_camera.initialize.return_value = True
        mock_camera_class.return_value = mock_camera
        
        mock_detector = Mock()
        mock_detector.initialize.return_value = True
        mock_detector_class.return_value = mock_detector
        
        mock_server = Mock()
        mock_server.run.side_effect = KeyboardInterrupt()
        mock_server_class.return_value = mock_server
        
        result = self.app.run()
        
        self.assertEqual(result, 0)
        mock_camera.release.assert_called_once()
        mock_detector.release.assert_called_once()
    
    @patch('src.main.PedestrianDetector')
    @patch('src.main.WebServer')
    @patch('src.main.CameraCapture')
    def test_run_exception(self, mock_camera_class, mock_server_class, mock_detector_class):
        """Test application run with exception."""
        mock_camera = Mock()
        mock_camera.initialize.return_value = True
        mock_camera_class.return_value = mock_camera
        
        mock_detector = Mock()
        mock_detector.initialize.return_value = True
        mock_detector_class.return_value = mock_detector
        
        mock_server = Mock()
        mock_server.run.side_effect = Exception("Test error")
        mock_server_class.return_value = mock_server
        
        result = self.app.run()
        
        self.assertEqual(result, 1)
        mock_camera.release.assert_called_once()
        mock_detector.release.assert_called_once()
    
    def test_cleanup_no_camera(self):
        """Test cleanup when camera is not initialized."""
        self.app.cleanup()
        # Should not raise exception
        self.assertIsNone(self.app.camera)
        self.assertIsNone(self.app.detector)
    
    @patch('src.main.PedestrianDetector')
    @patch('src.main.CameraCapture')
    def test_cleanup_with_resources(self, mock_camera_class, mock_detector_class):
        """Test cleanup with initialized resources."""
        mock_camera = Mock()
        mock_camera.initialize.return_value = True
        mock_camera_class.return_value = mock_camera
        
        mock_detector = Mock()
        mock_detector.initialize.return_value = True
        mock_detector_class.return_value = mock_detector
        
        self.app.initialize_camera()
        self.app.initialize_detector()
        self.app.cleanup()
        
        mock_camera.release.assert_called_once()
        mock_detector.release.assert_called_once()


class TestMainFunction(unittest.TestCase):
    """Unit tests for main function."""
    
    @patch('src.main.Application')
    def test_main_default_args(self, mock_app_class):
        """Test main function with default arguments."""
        mock_app = Mock()
        mock_app.run.return_value = 0
        mock_app_class.return_value = mock_app
        
        with patch.object(sys, 'argv', ['main.py']):
            result = main()
        
        self.assertEqual(result, 0)
        mock_app_class.assert_called_once_with(camera_id=0, host='0.0.0.0', port=8080)
        mock_app.run.assert_called_once()
    
    @patch('src.main.Application')
    def test_main_with_camera_id(self, mock_app_class):
        """Test main function with camera ID argument."""
        mock_app = Mock()
        mock_app.run.return_value = 0
        mock_app_class.return_value = mock_app
        
        with patch.object(sys, 'argv', ['main.py', '1']):
            result = main()
        
        mock_app_class.assert_called_once_with(camera_id=1, host='0.0.0.0', port=8080)
    
    @patch('src.main.Application')
    def test_main_with_all_args(self, mock_app_class):
        """Test main function with all arguments."""
        mock_app = Mock()
        mock_app.run.return_value = 0
        mock_app_class.return_value = mock_app
        
        with patch.object(sys, 'argv', ['main.py', '2', '127.0.0.1', '9090']):
            result = main()
        
        mock_app_class.assert_called_once_with(
            camera_id=2, 
            host='127.0.0.1', 
            port=9090
        )
    
    def test_main_invalid_camera_id(self):
        """Test main function with invalid camera ID."""
        with patch.object(sys, 'argv', ['main.py', 'invalid']):
            with patch('builtins.print') as mock_print:
                result = main()
        
        self.assertEqual(result, 1)
        mock_print.assert_called_once()
    
    def test_main_invalid_port(self):
        """Test main function with invalid port."""
        with patch.object(sys, 'argv', ['main.py', '0', '127.0.0.1', 'invalid']):
            with patch('builtins.print') as mock_print:
                result = main()
        
        self.assertEqual(result, 1)
        mock_print.assert_called_once()


if __name__ == '__main__':
    unittest.main()
