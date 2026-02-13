"""Integration tests for pedestrian detector module.

Tests detector with mocked and un-mocked external dependencies.
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import numpy as np

from src.detector import PedestrianDetector, DetectionResult


class TestPedestrianDetectorIntegration(unittest.TestCase):
    """Integration tests for PedestrianDetector with mocked dependencies."""
    
    @patch('src.detector.YOLO')
    def test_initialize_with_mock_yolo(self, mock_yolo_class):
        """Test initialization with mocked YOLO."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        detector = PedestrianDetector(use_tensorrt=False)
        result = detector.initialize()
        
        self.assertTrue(result)
        mock_yolo_class.assert_called_once_with('yolov8n.pt')
        self.assertEqual(detector.model, mock_model)
    
    @patch('src.detector.YOLO')
    def test_initialize_with_tensorrt_mock(self, mock_yolo_class):
        """Test initialization with mocked TensorRT export."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        detector = PedestrianDetector(use_tensorrt=True)
        result = detector.initialize()
        
        self.assertTrue(result)
        # Should try to export to TensorRT
        mock_model.export.assert_called_once_with(format='engine', half=True)
    
    @patch('src.detector.YOLO')
    def test_initialize_tensorrt_fallback_mock(self, mock_yolo_class):
        """Test TensorRT fallback when export fails."""
        mock_model = MagicMock()
        mock_model.export.side_effect = Exception("TensorRT not available")
        mock_yolo_class.return_value = mock_model
        
        detector = PedestrianDetector(use_tensorrt=True)
        result = detector.initialize()
        
        # Should still succeed with fallback
        self.assertTrue(result)
        # Should have attempted export
        mock_model.export.assert_called_once()
    
    @patch('src.detector.YOLO')
    def test_initialize_failure_mock(self, mock_yolo_class):
        """Test initialization failure with mocked YOLO."""
        mock_yolo_class.side_effect = Exception("Model not found")
        
        detector = PedestrianDetector()
        result = detector.initialize()
        
        self.assertFalse(result)
    
    @patch('src.detector.YOLO')
    def test_detect_with_mock_results(self, mock_yolo_class):
        """Test detection with mocked YOLO results."""
        # Create mock model
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        # Create mock detection results
        mock_box = MagicMock()
        mock_box.cls = [0]  # Person class
        mock_box.conf = [0.95]  # High confidence
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = [100, 100, 200, 300]
        
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        
        mock_model.return_value = [mock_result]
        
        # Initialize detector
        detector = PedestrianDetector(use_tensorrt=False)
        detector.initialize()
        
        # Run detection
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        # Verify results
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].confidence, 0.95)
        self.assertEqual(detections[0].bbox, (100.0, 100.0, 200.0, 300.0))
        
        # Verify model was called
        mock_model.assert_called_once()
    
    @patch('src.detector.YOLO')
    def test_detect_filters_low_confidence_mock(self, mock_yolo_class):
        """Test detection filters out low confidence detections."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        # Create mock low confidence detection
        mock_box = MagicMock()
        mock_box.cls = [0]  # Person class
        mock_box.conf = [0.3]  # Low confidence
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = [100, 100, 200, 300]
        
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        
        mock_model.return_value = [mock_result]
        
        # Initialize with 0.5 threshold
        detector = PedestrianDetector(confidence_threshold=0.5, use_tensorrt=False)
        detector.initialize()
        
        # Run detection
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        # Should be filtered out
        self.assertEqual(len(detections), 0)
    
    @patch('src.detector.YOLO')
    def test_detect_filters_non_person_class_mock(self, mock_yolo_class):
        """Test detection filters out non-person classes."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        # Create mock car detection (class 2)
        mock_box = MagicMock()
        mock_box.cls = [2]  # Car class
        mock_box.conf = [0.95]
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = [100, 100, 200, 300]
        
        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        
        mock_model.return_value = [mock_result]
        
        # Initialize detector
        detector = PedestrianDetector(use_tensorrt=False)
        detector.initialize()
        
        # Run detection
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        
        # Should be filtered out (not a person)
        self.assertEqual(len(detections), 0)


class TestPedestrianDetectorUnmocked(unittest.TestCase):
    """Integration tests without mocking to test real behavior."""
    
    def test_calculate_location_all_positions(self):
        """Test location calculation for all 9 positions."""
        detector = PedestrianDetector()
        frame_shape = (480, 640, 3)
        
        # Test all 9 combinations
        test_cases = [
            # Left column
            ((50, 400), "Left-Near"),
            ((50, 240), "Left-Mid"),
            ((50, 50), "Left-Far"),
            # Center column
            ((320, 400), "Center-Near"),
            ((320, 240), "Center-Mid"),
            ((320, 50), "Center-Far"),
            # Right column
            ((590, 400), "Right-Near"),
            ((590, 240), "Right-Mid"),
            ((590, 50), "Right-Far"),
        ]
        
        for (cx, cy), expected_location in test_cases:
            # Create bbox centered at (cx, cy)
            bbox = (cx - 25, cy - 50, cx + 25, cy + 50)
            location = detector._calculate_location(bbox, frame_shape)
            self.assertEqual(location, expected_location, 
                           f"Failed for center ({cx}, {cy})")
    
    def test_detection_result_attributes(self):
        """Test DetectionResult stores attributes correctly."""
        bbox = (10.5, 20.7, 100.3, 200.9)
        confidence = 0.87654
        location = "Right-Near"
        
        result = DetectionResult(bbox, confidence, location)
        
        self.assertEqual(result.bbox, bbox)
        self.assertAlmostEqual(result.confidence, confidence)
        self.assertEqual(result.location, location)
    
    def test_detection_result_to_dict(self):
        """Test DetectionResult to_dict method."""
        bbox = (10.5, 20.7, 100.3, 200.9)
        confidence = 0.87654
        location = "Right-Near"
        
        result = DetectionResult(bbox, confidence, location)
        data = result.to_dict()
        
        self.assertIsInstance(data, dict)
        self.assertIn('bbox', data)
        self.assertIn('confidence', data)
        self.assertIn('location', data)
        self.assertEqual(data['bbox']['x1'], 10.5)
        self.assertEqual(data['bbox']['x2'], 100.3)
        self.assertEqual(data['location'], location)


if __name__ == '__main__':
    unittest.main()
