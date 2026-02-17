"""Unit tests for pedestrian detector module."""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from src.detector import PedestrianDetector, DetectionResult


class TestDetectionResult(unittest.TestCase):
    """Test DetectionResult class."""
    
    def test_init(self):
        """Test DetectionResult initialization."""
        bbox = (10.0, 20.0, 100.0, 200.0)
        confidence = 0.95
        location = "Center-Mid"
        
        result = DetectionResult(bbox, confidence, location)
        
        self.assertEqual(result.bbox, bbox)
        self.assertEqual(result.confidence, confidence)
        self.assertEqual(result.location, location)
    
    def test_to_dict(self):
        """Test DetectionResult to_dict conversion."""
        bbox = (10.5, 20.7, 100.3, 200.9)
        confidence = 0.95678
        location = "Center-Mid"
        
        result = DetectionResult(bbox, confidence, location)
        data = result.to_dict()
        
        self.assertEqual(data['bbox']['x1'], 10.5)
        self.assertEqual(data['bbox']['y1'], 20.7)
        self.assertEqual(data['bbox']['x2'], 100.3)
        self.assertEqual(data['bbox']['y2'], 200.9)
        self.assertEqual(data['confidence'], 0.957)  # Rounded to 3 places
        self.assertEqual(data['location'], "Center-Mid")


class TestPedestrianDetector(unittest.TestCase):
    """Test PedestrianDetector class."""
    
    def test_init_default(self):
        """Test detector initialization with default parameters."""
        detector = PedestrianDetector()
        
        self.assertEqual(detector.model_path, 'yolov8n.pt')
        self.assertEqual(detector.confidence_threshold, 0.5)
        self.assertTrue(detector.use_tensorrt)
        self.assertIsNone(detector.model)
        self.assertEqual(detector.camera_angle, 60.0)
    
    def test_init_custom(self):
        """Test detector initialization with custom parameters."""
        detector = PedestrianDetector(
            model_path='custom.pt',
            confidence_threshold=0.7,
            use_tensorrt=False
        )
        
        self.assertEqual(detector.model_path, 'custom.pt')
        self.assertEqual(detector.confidence_threshold, 0.7)
        self.assertFalse(detector.use_tensorrt)
    
    def test_calculate_location_left_near(self):
        """Test location calculation for left-near position."""
        detector = PedestrianDetector()
        
        # Frame shape: 480x640
        frame_shape = (480, 640, 3)
        
        # Bbox in left-near region (bottom-left)
        bbox = (50.0, 400.0, 150.0, 470.0)
        
        location = detector._calculate_location(bbox, frame_shape)
        
        self.assertEqual(location, "Left-Near")
    
    def test_calculate_location_center_mid(self):
        """Test location calculation for center-mid position."""
        detector = PedestrianDetector()
        
        frame_shape = (480, 640, 3)
        
        # Bbox in center-mid region
        bbox = (250.0, 200.0, 390.0, 300.0)
        
        location = detector._calculate_location(bbox, frame_shape)
        
        self.assertEqual(location, "Center-Mid")
    
    def test_calculate_location_right_far(self):
        """Test location calculation for right-far position."""
        detector = PedestrianDetector()
        
        frame_shape = (480, 640, 3)
        
        # Bbox in right-far region (top-right)
        bbox = (500.0, 50.0, 630.0, 150.0)
        
        location = detector._calculate_location(bbox, frame_shape)
        
        self.assertEqual(location, "Right-Far")
    
    def test_detect_raises_when_no_model(self):
        """Test detect raises RuntimeError when model not initialized."""
        detector = PedestrianDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            detector.detect(frame)
    
    def test_draw_detections_empty(self):
        """Test drawing with no detections."""
        detector = PedestrianDetector()
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = []
        
        result = detector.draw_detections(frame, detections)
        
        # Should return unchanged frame
        np.testing.assert_array_equal(result, frame)
    
    def test_draw_detections_with_boxes(self):
        """Test drawing with detections."""
        detector = PedestrianDetector()
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            DetectionResult((100.0, 100.0, 200.0, 300.0), 0.95, "Center-Mid")
        ]
        
        result = detector.draw_detections(frame, detections)
        
        # Result should be different from original (has drawings)
        self.assertFalse(np.array_equal(result, frame))
        # Frame should not be modified in place
        np.testing.assert_array_equal(frame, np.zeros((480, 640, 3), dtype=np.uint8))
    
    def test_release(self):
        """Test release method."""
        detector = PedestrianDetector()
        detector.model = Mock()
        
        detector.release()
        
        self.assertIsNone(detector.model)


if __name__ == '__main__':
    unittest.main()
