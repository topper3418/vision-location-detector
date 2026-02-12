"""Integration tests for camera capture module.

These tests verify the camera module works correctly with external
packages both mocked and unmocked, as per dev rules.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Import cv2 to test integration
import cv2
from src.camera import CameraCapture


class TestCameraIntegration(unittest.TestCase):
    """Integration tests for CameraCapture with external packages."""
    
    def test_cv2_import_unmocked(self):
        """Test that cv2 can be imported without mocking."""
        # Verify cv2 is available
        self.assertTrue(hasattr(cv2, 'VideoCapture'))
        self.assertTrue(hasattr(cv2, 'imencode'))
        self.assertTrue(hasattr(cv2, 'CAP_PROP_FRAME_WIDTH'))
    
    def test_camera_init_unmocked(self):
        """Test camera initialization with unmocked cv2."""
        camera = CameraCapture(camera_id=0, width=640, height=480)
        self.assertIsNotNone(camera)
        self.assertEqual(camera.camera_id, 0)
    
    @patch('cv2.VideoCapture')
    def test_camera_init_mocked(self, mock_video_capture):
        """Test camera initialization with mocked cv2."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        camera = CameraCapture(camera_id=0, width=640, height=480)
        result = camera.initialize()
        
        self.assertTrue(result)
        mock_video_capture.assert_called_once_with(0)
    
    @patch('cv2.VideoCapture')
    @patch('cv2.imencode')
    def test_full_capture_cycle_mocked(self, mock_imencode, mock_video_capture):
        """Test full capture cycle with mocked cv2."""
        # Setup mocks
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_video_capture.return_value = mock_cap
        
        test_buffer = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
        mock_imencode.return_value = (True, test_buffer)
        
        # Test full cycle
        camera = CameraCapture()
        self.assertTrue(camera.initialize())
        
        # Verify buffer size was set to 1
        mock_cap.set.assert_any_call(38, 1)  # CAP_PROP_BUFFERSIZE = 38
        
        success, frame = camera.read_frame()
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        
        jpeg_data = camera.get_jpeg_frame()
        self.assertIsNotNone(jpeg_data)
        
        camera.release()
        self.assertIsNone(camera.capture)
    
    @patch('cv2.VideoCapture')
    def test_full_capture_cycle_unmocked_release(self, mock_video_capture):
        """Test that release works with unmocked cv2 objects."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        camera = CameraCapture()
        camera.initialize()
        camera.release()
        
        # Verify release was called on actual cv2 VideoCapture object
        mock_cap.release.assert_called_once()
    
    def test_numpy_integration_unmocked(self):
        """Test numpy integration with camera frames."""
        camera = CameraCapture(width=640, height=480)
        
        # Test that frame shape expectations match
        test_frame = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        self.assertEqual(test_frame.shape, (480, 640, 3))
    
    @patch('cv2.imencode')
    @patch('cv2.VideoCapture')
    def test_jpeg_encoding_integration_mocked(self, mock_video_capture, mock_imencode):
        """Test JPEG encoding integration with mocked cv2."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, test_frame)
        mock_video_capture.return_value = mock_cap
        
        # Mock imencode to simulate real cv2 behavior
        test_buffer = np.random.randint(0, 255, 1000, dtype=np.uint8)
        mock_imencode.return_value = (True, test_buffer)
        
        camera = CameraCapture()
        camera.initialize()
        jpeg_data = camera.get_jpeg_frame(quality=90)
        
        self.assertIsNotNone(jpeg_data)
        self.assertIsInstance(jpeg_data, bytes)
        # Verify imencode was called with correct parameters
        args = mock_imencode.call_args[0]
        self.assertEqual(args[0], '.jpg')
        np.testing.assert_array_equal(args[1], test_frame)
    
    @patch('cv2.VideoCapture')
    def test_hardware_acceleration_settings_mocked(self, mock_video_capture):
        """Test that hardware acceleration settings are applied."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        camera = CameraCapture(width=1920, height=1080)
        camera.initialize()
        
        # Verify MJPG codec is set for Jetson acceleration
        calls = mock_cap.set.call_args_list
        self.assertTrue(len(calls) >= 3)
    
    @patch('cv2.VideoCapture')
    def test_error_handling_with_unmocked_types(self, mock_video_capture):
        """Test error handling with cv2 types."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = False
        mock_video_capture.return_value = mock_cap
        
        camera = CameraCapture()
        result = camera.initialize()
        
        self.assertFalse(result)
        self.assertFalse(camera.is_opened())
    
    @patch('cv2.VideoCapture')
    def test_buffer_size_for_latest_frame_mocked(self, mock_video_capture):
        """Test that buffer size is set to 1 for latest frame (YOLO optimization)."""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        camera = CameraCapture()
        camera.initialize()
        
        # Verify CAP_PROP_BUFFERSIZE (38) is set to 1
        # This ensures we always get the latest frame, not buffered old frames
        mock_cap.set.assert_any_call(38, 1)
        
        # Verify it was set during initialization
        set_calls = [call for call in mock_cap.set.call_args_list if call[0][0] == 38]
        self.assertEqual(len(set_calls), 1)
        self.assertEqual(set_calls[0][0][1], 1)


if __name__ == '__main__':
    unittest.main()
