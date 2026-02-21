
# Vision Location Detector

Vision Location Detector is a real-time video streaming and pedestrian detection application. It is optimized for Nvidia Jetson Orin Nano but runs on any Linux system with Python 3.8+. The app captures camera footage, performs optional YOLOv8-based pedestrian detection, and streams results to a web interface.

Configuration is managed via a `.env` file. The server is built with aiohttp (async web framework) and OpenCV for camera handling.


## Features

- **Real-time Video Streaming** (MJPEG, ~30 FPS)
- **Optional Pedestrian Detection** (YOLOv8, configurable)
- **Web Interface** (HTML, live detection overlay)
- **Hardware Optimized** (TensorRT for Jetson, CUDA support)
- **Async Server** (aiohttp)
- **Configurable via .env** (device, camera, detection, server, performance)
- **REST API** (`/detections`, `/fps`, `/settings`)


## Project Structure

```
vision-location-detector/
├── src/
│   ├── __init__.py
│   ├── main.py                # Main entry point
│   ├── server.py              # Web server (aiohttp)
│   ├── server_builder.py      # Server and pipeline builder
│   ├── settings.py            # Loads .env and environment variables
│   ├── static/
│   │   └── index.html         # Web interface
│   ├── detection_services/
│   │   └── pedestrian_detector.py  # YOLOv8 pedestrian detector
│   ├── interfaces/
│   │   ├── detection_result.py
│   │   ├── detector_delegate.py
│   │   └── video_feed_base.py
│   ├── util/
│   │   └── get_video_path.py
│   └── video_feeds/
│       ├── camera_feed.py
│       └── spoof_video_feed.py
├── scripts/                  # Diagnostic and utility scripts
│   ├── check_compatibility.py
│   ├── detect_on_video.py
│   ├── detect_on_video_light.py
│   ├── detect_on_video_v2.py
│   ├── diagnose_pytorch.py
│   ├── kill_server.py
│   ├── record_video.py
│   ├── test_engine_initialization.py
│   └── test_installation.py
├── test/
│   ├── unit/
│   │   ├── test_camera.py
│   │   ├── test_detector.py
│   │   ├── test_main.py
│   │   └── test_server.py
│   └── integration/
│       ├── test_camera_integration.py
│       ├── test_detector_integration.py
│       ├── test_main_integration.py
│       └── test_server_integration.py
├── captures/                 # Saved video captures
├── test_data/                # Test images/videos
├── .env                      # Main configuration file
├── CONFIG.md                 # Configuration guide
├── run.sh                    # Example run script
├── yolov8n.pt                # YOLOv8 model (nano)
├── yolov8n.engine            # TensorRT engine (auto-generated)
├── yolov8n.onnx              # ONNX model (optional)
└── README.md
```


## Development Rules

See `CONFIG.md` for full configuration and tuning options.

Key rules:
- All source code in `src/`
- Modules >250 lines should be refactored into packages
- One class per module (object-oriented)


## Installation & Setup

### Requirements
- Python 3.8+
- Camera device (USB or CSI)
- For Jetson: CUDA-enabled OpenCV, TensorRT (optional)

### Setup
1. Clone the repository:
	```bash
	git clone <repo-url>
	cd vision-location-detector
	```
2. (Recommended) Create and activate a virtual environment:
	```bash
	python3 -m venv venv
	source venv/bin/activate
	```
3. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
4. Copy and edit `.env` for configuration:
	```bash
	cp example.env .env
	# Edit .env as needed (see CONFIG.md for options)
	```


## Usage

### Running the Application

Start the server (loads settings from `.env`):
```bash
python -m src.main
```

All configuration (device, detection, camera, server, performance) is controlled via `.env`.


### Debugging

Use VS Code launch configurations or run scripts in the `scripts/` directory for diagnostics and testing.


### Stopping the Server

Kill the running server:
```bash
python scripts/kill_server.py
```


### Accessing the Web Interface

Open a browser and go to:
```
http://localhost:8080
```
Or from another device:
```
http://<server-ip>:8080
```


## Testing

Run all tests:
```bash
python -m unittest discover -s test
```
Run only unit or integration tests:
```bash
python -m unittest discover -s test/unit
python -m unittest discover -s test/integration
```
Run a specific test module:
```bash
python -m unittest test.unit.test_camera
```


## API Endpoints

- `GET /` — HTML web interface
- `GET /stream` — MJPEG video stream
- `GET /detections` — JSON with current detection results
- `GET /fps` — JSON with current FPS
- `GET /settings` — JSON with current server settings


## Hardware & Performance

- **Camera Angle:** Recommended 60° downward for optimal floor coverage
- **Jetson Optimization:**
	- MJPEG codec for hardware acceleration
	- TensorRT acceleration (set `USE_TENSORRT=true` in `.env`)
	- OpenCV with CUDA support
	- Lower JPEG quality for higher FPS


## Configuration

All settings are controlled via `.env`. See `CONFIG.md` for full documentation and tuning tips.


## License

Copyright © 2026

---

**Technical Details:**
- Framework: aiohttp (async web framework)
- Camera: OpenCV (cv2)
- Detection: YOLOv8 (ultralytics)
- Video: MJPEG streaming
- Default resolution: 640x480 (configurable)
- All configuration: `.env` file
