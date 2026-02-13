"""Application settings management.

Loads configuration from environment variables and .env file.
"""

import os
from typing import Optional
from pathlib import Path


class Settings:
    """Application settings loaded from environment."""
    
    def __init__(self):
        """Initialize settings from environment variables."""
        # Load .env file if it exists
        self._load_env_file()
        
        # Device Configuration
        self.device: str = os.getenv('DEVICE', 'cpu')
        
        # YOLO Configuration
        self.enable_yolo: bool = os.getenv('ENABLE_YOLO', 'false').lower() == 'true'
        self.use_tensorrt: bool = os.getenv('USE_TENSORRT', 'true').lower() == 'true'
        self.yolo_model_path: str = os.getenv('YOLO_MODEL_PATH', 'yolov8n.pt')
        self.confidence_threshold: float = float(os.getenv('CONFIDENCE_THRESHOLD', '0.5'))
        
        # Camera Configuration
        self.camera_id: int = int(os.getenv('CAMERA_ID', '0'))
        self.camera_width: int = int(os.getenv('CAMERA_WIDTH', '640'))
        self.camera_height: int = int(os.getenv('CAMERA_HEIGHT', '480'))
        
        # Server Configuration
        self.server_host: str = os.getenv('SERVER_HOST', '0.0.0.0')
        self.server_port: int = int(os.getenv('SERVER_PORT', '8080'))
        
        # Performance Settings
        self.jpeg_quality: int = int(os.getenv('JPEG_QUALITY', '70'))
    
    def _load_env_file(self) -> None:
        """Load environment variables from .env file if it exists."""
        env_path = Path(__file__).parent.parent / '.env'
        
        if not env_path.exists():
            return
        
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Only set if not already in environment
                    if key and not os.getenv(key):
                        os.environ[key] = value
    
    def __repr__(self) -> str:
        """String representation of settings."""
        return (
            f"Settings("
            f"device={self.device}, "
            f"enable_yolo={self.enable_yolo}, "
            f"camera={self.camera_id}, "
            f"resolution={self.camera_width}x{self.camera_height}, "
            f"server={self.server_host}:{self.server_port})"
        )


# Global settings instance
settings = Settings()
