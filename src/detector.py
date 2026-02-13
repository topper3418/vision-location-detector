"""Pedestrian detection module using YOLO.

This module handles pedestrian detection in video frames using YOLOv8,
optimized for Nvidia Jetson Orin Nano hardware with TensorRT acceleration.
"""

from typing import List, Tuple, Optional
import numpy as np
from ultralytics import YOLO


class DetectionResult:
    """Represents a single pedestrian detection."""
    
    def __init__(self, bbox: Tuple[float, float, float, float], 
                 confidence: float, location: str):
        """Initialize detection result.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2) in pixels
            confidence: Detection confidence score (0-1)
            location: Approximate location description
        """
        self.bbox = bbox
        self.confidence = confidence
        self.location = location
    
    def to_dict(self) -> dict:
        """Convert detection result to dictionary.
        
        Returns:
            Dictionary representation of the detection
        """
        return {
            'bbox': {
                'x1': self.bbox[0],
                'y1': self.bbox[1],
                'x2': self.bbox[2],
                'y2': self.bbox[3]
            },
            'confidence': round(self.confidence, 3),
            'location': self.location
        }


class PedestrianDetector:
    """Handles pedestrian detection using YOLO."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', 
                 confidence_threshold: float = 0.5,
                 use_tensorrt: bool = True,
                 device: Optional[str] = None):
        """Initialize pedestrian detector.
        
        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections (0-1)
            use_tensorrt: Enable TensorRT acceleration for Jetson
            device: Device to run inference on ('0' for GPU, 'cpu' for CPU, None for auto-detect)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.use_tensorrt = use_tensorrt
        
        # Auto-detect device if not specified
        if device is None:
            try:
                import torch
                self.device = '0' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                self.device = 'cpu'
        else:
            self.device = device
            
        self.model: Optional[YOLO] = None
        self.camera_angle: float = 60.0  # degrees downward
        
    def initialize(self) -> bool:
        """Initialize the YOLO model.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.model = YOLO(self.model_path)
            
            # Export to TensorRT for Jetson acceleration if requested
            if self.use_tensorrt and self.model is not None:
                try:
                    # Try to export to TensorRT format
                    # On Jetson, this will use GPU acceleration
                    self.model.export(format='engine', half=True)
                    # Load the TensorRT model
                    engine_path = self.model_path.replace('.pt', '.engine')
                    self.model = YOLO(engine_path)
                except Exception as e:
                    # Fallback to regular model if TensorRT fails
                    # This is expected on non-Jetson hardware
                    print(f"TensorRT export failed, using default model: {e}")
                    self.model = YOLO(self.model_path)
            
            return True
        except Exception as e:
            print(f"Failed to initialize YOLO model: {e}")
            return False
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect pedestrians in a frame.
        
        Args:
            frame: Input image as numpy array (BGR format)
            
        Returns:
            List of DetectionResult objects
        """
        if self.model is None:
            return []
        
        # Run YOLO detection with optimized settings
        # Use half precision (FP16) only on GPU for speed
        # imgsz=640 for balanced speed/accuracy
        use_half = self.device != 'cpu'
        results = self.model(frame, verbose=False, device=self.device, imgsz=640, half=use_half)
        
        detections = []
        
        # Process results
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get class ID and filter for person class (class 0 in COCO)
                class_id = int(box.cls[0])
                if class_id != 0:  # 0 is the person class in COCO dataset
                    continue
                
                # Get confidence
                confidence = float(box.conf[0])
                if confidence < self.confidence_threshold:
                    continue
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                bbox = (float(x1), float(y1), float(x2), float(y2))
                
                # Calculate approximate location
                # Ensure frame.shape is a 3-tuple (height, width, channels)
                shape = frame.shape if len(frame.shape) == 3 else (frame.shape[0], frame.shape[1], 1)
                location = self._calculate_location(bbox, shape)
                
                detections.append(DetectionResult(bbox, confidence, location))
        
        return detections
    
    def _calculate_location(self, bbox: Tuple[float, float, float, float], 
                           frame_shape: Tuple[int, int, int]) -> str:
        """Calculate approximate location of detected person.
        
        Given that the camera is angled 60 degrees downward, we can estimate
        the person's location relative to the camera based on their position
        in the frame.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_shape: Frame dimensions (height, width, channels)
            
        Returns:
            String description of location (e.g., "Center-Near", "Left-Far")
        """
        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]
        
        # Calculate center of bounding box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Determine horizontal position (Left, Center, Right)
        if center_x < width / 3:
            horizontal = "Left"
        elif center_x < 2 * width / 3:
            horizontal = "Center"
        else:
            horizontal = "Right"
        
        # Determine depth/distance (Near, Mid, Far)
        # With camera angled downward, objects higher in frame are farther away
        if center_y > 2 * height / 3:
            depth = "Near"
        elif center_y > height / 3:
            depth = "Mid"
        else:
            depth = "Far"
        
        return f"{horizontal}-{depth}"
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[DetectionResult]) -> np.ndarray:
        """Draw detection boxes and labels on frame.
        
        Args:
            frame: Input image as numpy array
            detections: List of DetectionResult objects
            
        Returns:
            Frame with drawn detections
        """
        import cv2
        
        frame_copy = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence and location
            label = f"{detection.location} ({detection.confidence:.2f})"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                frame_copy,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return frame_copy
    
    def release(self) -> None:
        """Release resources."""
        # YOLO model doesn't require explicit cleanup
        self.model = None
