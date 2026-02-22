"""Pedestrian detection module using YOLO.

This module handles pedestrian detection in video frames using YOLOv8,
optimized for Nvidia Jetson Orin Nano hardware with TensorRT acceleration.
"""

from typing import List, Tuple, Optional
import numpy as np
from ultralytics import YOLO

from src.interfaces.detector_delegate import DetectorDelegate
from src.interfaces.detection_result import DetectionResult


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
            
            if self.use_tensorrt:
                if self.device != 'cuda':
                    raise RuntimeError(
                        "TensorRT acceleration requested, but CUDA is not available. "
                        "Cannot proceed without GPU. Aborting."
                    )
                
                # Double-check CUDA (redundant but explicit)
                import torch
                if not (hasattr(torch, 'cuda') and torch.cuda.is_available()):
                    raise RuntimeError(
                        "TensorRT acceleration is requested, but no CUDA device is available. "
                        "Cannot continue without GPU acceleration."
                    )
                
                import os
                engine_path = self.model_path.replace('.pt', '.engine')
                
                if not os.path.exists(engine_path):
                    print("TensorRT engine not found. Exporting with FP16 (this may take several minutes)...")
                    exported = self.model.export(
                        format='engine',
                        half=True,                # FP16 mandatory for best Jetson performance
                        device=0,                 # Explicit GPU (index 0)
                        imgsz=640,                # Fixed input size; change if your camera needs different
                        workspace=4,              # 4 GB workspace — increase to 6–8 if you get OOM
                        simplify=True,            # ONNX slimming (usually safe and beneficial)
                        verbose=True              # Helpful debug output during export
                    )
                    print(f"TensorRT engine exported to: {exported}")
                    engine_path = exported  # Ultralytics returns the actual path used
                
                # Load the engine — this must succeed
                self.model = YOLO(engine_path, task='detect', verbose=False)
                print("TensorRT engine loaded successfully")
            
            # No else block needed — if not use_tensorrt, we just use the .pt model on whatever device was selected
            # (You can add a print here if you want visibility)
            if not self.use_tensorrt:
                print(f"TensorRT disabled — using native model on {self.device}")
            
            print(f"YOLO model initialized successfully on {self.device}")
            return True
        
        except Exception as e:
            print(f"YOLO initialization failed: {e}")
            self.model = None
            return False
        
        except Exception as e:
            print(f"YOLO initialization failed: {e}")
            self.model = None
            return False
    
    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Detect pedestrians in a frame and return DetectionResult list."""
        if self.model is None:
            raise RuntimeError("YOLO model is not initialized. Aborting.")
        results = self.model(
            frame, 
            verbose=False,
            classes=[0],  # Only detect person class (class 0)
            conf=self.confidence_threshold,
            )
        detections: List[DetectionResult] = []
        for result in results:
            for box in getattr(result, 'boxes', []):
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