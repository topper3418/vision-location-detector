"""Unit tests for camera capture module."""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.camera import CameraCapture


class TestCameraCapture(unittest.TestCase):
    """Unit tests for CameraCapture class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.camera = CameraCapture(camera_id=0, width=640, height=480)
    
    def test_init(self):
        """Test camera initialization parameters."""
        self.assertEqual(self.camera.camera_id, 0)
        self.assertEqual(self.camera.width, 640)
        self.assertEqual(self.camera.height, 480)
        self.assertIsNone(self.camera.capture)
    
    def test_init_custom_params(self):
        """Test camera initialization with custom parameters."""
        camera = CameraCapture(camera_id=1, width=1920, height=1080)
        self.assertEqual(camera.camera_id, 1)
        self.assertEqual(camera.width, 1920)
        self.assertEqual(camera.height, 1080)
    
    @patch('cv2.VideoCapture')
    def test_initialize_success(self, mock_video_capture):
        """Test successful camera initialization."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        result = self.camera.initialize()
        
        self.assertTrue(result)
        mock_video_capture.assert_called_once_with(0)
        mock_cap.set.assert_any_call(3, 640)  # CAP_PROP_FRAME_WIDTH = 3
        mock_cap.set.assert_any_call(4, 480)  # CAP_PROP_FRAME_HEIGHT = 4
    
    @patch('cv2.VideoCapture')
    def test_initialize_failure(self, mock_video_capture):
        """Test failed camera initialization."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        result = self.camera.initialize()
        
        self.assertFalse(result)
    
    def test_read_frame_no_capture(self):
        """Test reading frame when camera not initialized."""
        success, frame = self.camera.read_frame()
        
        self.assertFalse(success)
        self.assertIsNone(frame)
    
    @patch('cv2.VideoCapture')
    def test_read_frame_success(self, mock_video_capture):
        """Test successful frame reading with buffer flush."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_video_capture.return_value = mock_cap
        
        self.camera.initialize()
        success, frame = self.camera.read_frame()
        
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        np.testing.assert_array_equal(frame, test_frame)
        
        # Verify buffer was flushed (grab called 4 times)
        self.assertEqual(mock_cap.grab.call_count, 4)
        mock_cap.read.assert_called_once()
    
    @patch('cv2.VideoCapture')
    def test_read_frame_failure(self, mock_video_capture):
        """Test failed frame reading."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_video_capture.return_value = mock_cap
        
        self.camera.initialize()
        success, frame = self.camera.read_frame()
        
        self.assertFalse(success)
        self.assertIsNone(frame)
    
    @patch('cv2.VideoCapture')
    def test_read_frame_buffer_flush(self, mock_video_capture):
        """Test that buffer is flushed to get latest frame."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_cap.grab.return_value = True
        mock_video_capture.return_value = mock_cap
        
        self.camera.initialize()
        
        # Read a frame
        success, frame = self.camera.read_frame()
        
        # Verify grab was called exactly 4 times to flush buffer
        self.assertEqual(mock_cap.grab.call_count, 4)
        # Verify read was called once after flushing
        mock_cap.read.assert_called_once()
        self.assertTrue(success)
    
    def test_get_jpeg_frame_no_capture(self):
        """Test getting JPEG frame when camera not initialized."""
        result = self.camera.get_jpeg_frame()
        
        self.assertIsNone(result)
    
    @patch('cv2.imencode')
    @patch('cv2.VideoCapture')
    def test_get_jpeg_frame_success(self, mock_video_capture, mock_imencode):
        """Test successful JPEG frame encoding."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_video_capture.return_value = mock_cap
        
        test_buffer = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        mock_imencode.return_value = (True, test_buffer)
        
        self.camera.initialize()
        result = self.camera.get_jpeg_frame(quality=85)
        
        self.assertIsNotNone(result)
        self.assertEqual(result, test_buffer.tobytes())
        mock_imencode.assert_called_once()
    
    @patch('cv2.imencode')
    @patch('cv2.VideoCapture')
    def test_get_jpeg_frame_encode_failure(self, mock_video_capture, mock_imencode):
        """Test JPEG encoding failure."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_video_capture.return_value = mock_cap
        
        mock_imencode.return_value = (False, None)
        
        self.camera.initialize()
        result = self.camera.get_jpeg_frame()
        
        self.assertIsNone(result)
    
    @patch('cv2.VideoCapture')
    def test_release(self, mock_video_capture):
        """Test camera release."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        self.camera.initialize()
        self.camera.release()
        
        mock_cap.release.assert_called_once()
        self.assertIsNone(self.camera.capture)
    
    def test_release_no_capture(self):
        """Test releasing when camera not initialized."""
        self.camera.release()
        # Should not raise an exception
        self.assertIsNone(self.camera.capture)
    
    @patch('cv2.VideoCapture')
    def test_is_opened_true(self, mock_video_capture):
        """Test is_opened returns True for opened camera."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        self.camera.initialize()
        
        self.assertTrue(self.camera.is_opened())
    
    def test_is_opened_false(self):
        """Test is_opened returns False for uninitialized camera."""
        self.assertFalse(self.camera.is_opened())
    
    @patch('cv2.VideoCapture')
    def test_is_opened_after_release(self, mock_video_capture):
        """Test is_opened returns False after release."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        self.camera.initialize()
        self.camera.release()
        
        self.assertFalse(self.camera.is_opened())


if __name__ == '__main__':
    unittest.main()
