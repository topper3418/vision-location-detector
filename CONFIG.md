# Configuration Guide

## Environment Variables (.env)

The application is now configured via the `.env` file and environment variables. Edit `.env` to change settings:

### Device & Performance
- `DEVICE`: Set to `cuda` or `0` for GPU, `cpu` for CPU (default: `cpu`)
- `JPEG_QUALITY`: JPEG encoding quality 1-100, lower = faster (default: `50`)

### YOLO Detection
- `ENABLE_YOLO`: Enable/disable YOLO detection (default: `false`)
  - Set to `true` to enable pedestrian detection
  - Set to `false` for fast video-only mode (50-70 FPS expected)
- `USE_TENSORRT`: Enable TensorRT acceleration on Jetson (default: `true`)
- `YOLO_MODEL_PATH`: Path to YOLO model file (default: `yolov8n.pt`)
- `CONFIDENCE_THRESHOLD`: Detection confidence threshold 0-1 (default: `0.5`)

### Camera
- `CAMERA_ID`: Camera device ID, usually `0` for primary camera
- `CAMERA_WIDTH`: Frame width in pixels (default: `640`)
- `CAMERA_HEIGHT`: Frame height in pixels (default: `480`)

### Server
- `SERVER_HOST`: Server bind address (default: `0.0.0.0`)
- `SERVER_PORT`: Server port (default: `8080`)

## Performance Tuning

### Maximum FPS (Video Only)
For fastest performance without detection:
```bash
ENABLE_YOLO=false
JPEG_QUALITY=50
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
```

### GPU-Accelerated Detection on Jetson
For best performance with YOLO on Jetson Orin Nano:
```bash
DEVICE=0               # or 'cuda'
ENABLE_YOLO=true
USE_TENSORRT=true
JPEG_QUALITY=70
```

### CPU-Only Mode
If CUDA is not available:
```bash
DEVICE=cpu
ENABLE_YOLO=false      # Recommended - CPU YOLO is very slow
JPEG_QUALITY=50
```

## Troubleshooting

### Low FPS (< 10 FPS without YOLO)
Possible causes:
1. **Camera hardware limitation** - Try lower resolution:
   ```bash
   CAMERA_WIDTH=320
   CAMERA_HEIGHT=240
   ```

2. **USB bandwidth** - Use a different USB port or camera

3. **JPEG encoding** - Lower quality:
   ```bash
   JPEG_QUALITY=30
   ```

### YOLO Too Slow
1. Ensure GPU is being used:
   ```bash
   DEVICE=0
   USE_TENSORRT=true
   ```

2. Use lighter model:
   ```bash
   YOLO_MODEL_PATH=yolov8n.pt  # nano - fastest
   # vs yolov8s.pt (small), yolov8m.pt (medium)
   ```

3. Disable YOLO for pure video feed:
   ```bash
   ENABLE_YOLO=false
   ```

## Starting the Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start server (loads settings from .env)
python -m src.main

# Or run in background
python -m src.main > server.log 2>&1 &
```

The server will log its configuration on startup showing which settings are active.
