# Vision Location Detector

A real-time video streaming application designed for Nvidia Jetson Orin Nano that captures camera footage and streams it to a web interface. Built with aiohttp for asynchronous web serving and OpenCV for camera handling.

## Features

- **Real-time Video Streaming**: MJPEG streaming at ~30 FPS
- **Pedestrian Detection**: YOLOv8-based pedestrian detection with location tracking
- **Web Interface**: Clean, responsive HTML interface with live detection updates
- **Hardware Optimized**: Configured for Nvidia Jetson Orin Nano with TensorRT acceleration
- **Asynchronous**: Built with aiohttp for efficient async request handling
- **Camera Positioning**: Designed for 60° downward angle for optimal floor coverage
- **REST API**: JSON endpoint for detection data

## Project Structure

```
vision-location-detector/
├── src/
│   ├── __init__.py
│   ├── camera.py          # Camera capture module
│   ├── server.py          # Web server module
│   ├── main.py            # Main entry point
│   └── static/
│       └── index.html     # Web interface
├── test/
│   ├── unit/              # Unit tests
│   │   ├── test_camera.py
│   │   ├── test_server.py
│   │   └── test_main.py
│   └── integration/       # Integration tests
│       ├── test_camera_integration.py
│       ├── test_server_integration.py
│       └── test_main_integration.py
├── server/                # Server configuration files
└── requirements.txt       # Python dependencies
```

## Development Rules

This project follows strict development rules:

1. **All functions must have unit tests** in the `test/unit` directory
2. **Functions importing external packages** must have mocking and un-mocking integration tests
3. **All source code** goes in the `src/` directory
4. **Server configuration files** (nginx, systemd, etc.) go in the `server/` directory
5. **Modules exceeding 250 lines** must be refactored into packages
6. **Object-oriented patterns** - one class per module

## Installation

### Requirements

- Python 3.8+
- Camera device (USB or CSI camera)
- For Jetson: CUDA-enabled OpenCV

### Setup

1. Clone the repository:
```bash
cd /home/travisopperud/vision-location-detector
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Basic usage (default camera, port 8080):
```bash
python -m src.main
```

Custom camera ID:
```bash
python -m src.main 1
```

Custom host and port:
```bash
python -m src.main 0 0.0.0.0 9090
```

### Debugging in VS Code

Use the provided launch configurations in `.vscode/launch.json`:
- **Python: Run Server** - Run with default settings
- **Python: Run Server (Custom Camera)** - Specify camera, host, and port
- **Python: Run Server (Port 9090)** - Run on alternate port
- **Python: Run Tests** - Run all tests with pytest

Press F5 or use the Run and Debug panel to start debugging.

### Stopping the Server

Kill the running server using either method:

**Bash script:**
```bash
./kill_server.sh
```

**Python script:**
```bash
python -m src.kill_server
# Or with force flag
python -m src.kill_server --force
```

### Accessing the Web Interface

Once the application is running, open a web browser and navigate to:
```
http://localhost:8080
```

Or from another device on the network:
```
http://<jetson-ip-address>:8080
```

## Testing

### Run All Tests

```bash
# Run all tests
python -m unittest discover -s test

# Run only unit tests
python -m unittest discover -s test/unit

# Run only integration tests
python -m unittest discover -s test/integration
```

### Run Specific Test Modules

```bash
# Test camera module
python -m unittest test.unit.test_camera

# Test server module
python -m unittest test.unit.test_server

# Test main module
python -m unittest test.unit.test_main
```

## API Endpoints

- `GET /` - Serves the HTML interface
- `GET /stream` - MJPEG video stream endpoint

## Hardware Configuration

### Camera Angle
The camera should be mounted at a **45° downward angle** for optimal floor coverage and person detection (future feature).

### Jetson Optimization
The application is optimized for Nvidia Jetson Orin Nano:
- MJPEG codec for hardware acceleration
- Efficient async I/O with aiohttp
- OpenCV configured for CUDA acceleration

## Future Development

This initial version provides the video feed webapp without object detection. Future phases will include:

1. YOLO integration for person detection
2. Bounding boxes around detected people
3. Location approximation and display
4. Detection list overlay on the web interface

## License

Copyright © 2026

## Technical Details

- **Framework**: aiohttp (async web framework)
- **Camera Library**: OpenCV (cv2)
- **Video Format**: MJPEG streaming
- **Frame Rate**: ~30 FPS
- **Default Resolution**: 640x480 (configurable)
