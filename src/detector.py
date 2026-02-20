"""Pedestrian detection module using YOLO.

This module handles pedestrian detection in video frames using YOLOv8,
optimized for Nvidia Jetson Orin Nano hardware with TensorRT acceleration.
"""

from typing import List, Tuple, Optional
import numpy as np
from ultralytics import YOLO

from .detector_delegate import DetectorDelegate


from .video_feed_base import DetectionResult


class PedestrianDetector(DetectorDelegate):
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
        # Device selection logic
        if device is None:
            # Auto-detect: use CUDA if available, else CPU
            try:
                import torch
                self.device = 'cuda' if (hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'cpu'
            except ImportError:
                self.device = 'cpu'
        else:
            # Normalize device names
            if device in ['cuda', 'gpu', '0']:
                import torch
                if not (hasattr(torch, 'cuda') and torch.cuda.is_available()):
                    raise RuntimeError("CUDA device requested but not available. Aborting.")
                self.device = 'cuda'
            elif device == 'cpu':
                self.device = 'cpu'
            else:
                self.device = device
            
        self.model: Optional[YOLO] = None
        self.camera_angle: float = 60.0  # degrees downward
    
    def initialize(self) -> bool:
        """Initialize the YOLO model. Returns False on failure."""
        print(f"Initializing YOLO model on device: {self.device}")
        try:
            self.model = YOLO(self.model_path, task='detect')
            # Export to TensorRT for Jetson acceleration if requested and CUDA is available
            if self.use_tensorrt and self.model is not None and self.device == 'cuda':
                import os
                engine_path = self.model_path.replace('.pt', '.engine')
                if os.path.exists(engine_path):
                    print(f"Found existing TensorRT engine: {engine_path}. Loading...")
                    self.model = YOLO(engine_path)
                    print("TensorRT model loaded successfully")
                else:
                    print("TensorRT engine not found. Exporting...")
                    self.model.export(format='engine', half=True, device=self.device)
                    self.model = YOLO(engine_path)
                    print("TensorRT model exported and loaded successfully")
            elif self.use_tensorrt and self.device != 'cuda':
                raise RuntimeError("TensorRT requested but CUDA not available. Aborting.")
            print(f"YOLO model initialized successfully on {self.device}")
            return True
        except Exception as e:
            print(f"YOLO initialization failed: {e}")
            self.model = None
            return False
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect pedestrians in a frame and return DetectionResult list."""
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized. Aborting.")
        results = self.model(frame)
        detections: List[DetectionResult] = []
        for result in results:
            for box in getattr(result, 'boxes', []):
                # Only keep person class (class 0)
                if hasattr(box, 'cls') and box.cls[0] != 0:
                    continue
                conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                if conf < self.confidence_threshold:
                    continue
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box, 'xyxy') else [0, 0, 0, 0]
                bbox = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
                detections.append(DetectionResult(bbox, conf, label="person"))
        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[DetectionResult]) -> np.ndarray:
        """Draw detection boxes and labels on frame."""
        import cv2
        frame_copy = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{detection.label} ({detection.confidence:.2f})"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                frame_copy,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
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
        """Release any resources held by the detector."""
        self.model = None