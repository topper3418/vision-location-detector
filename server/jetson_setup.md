# Initial Installation

The following instructions are for properly setting up the Jetson
Orin Nano to utilize its GPU

## Out of the box

TODO: fill this out once I've gotten a new one to test this out on. 
The Jetson must first be flashed with 5.x and then have an update or 
two installed. 

## Flashing with OS 6.2 

1. Install the ISO image from [this link](https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.3/jp62-orin-nano-sd-card-image.zip)

2. Use Balena Etcher to flash onto a suitable microSD card

3. Update the firmware
```bash
sudo apt update && sudo apt upgrade -y
```

## Installing global dependencies

Not all default pip repos are configured to properly utilize the jetson's
GPU to its fullest potential. These installs will greatly improve performance. 
The following is from [Ultralytic's quick start guide](https://docs.ultralytics.com/guides/nvidia-jetson/#install-pytorch-and-torchvision_1)

1. install and update pip

```bash
sudo apt update
sudo apt install python3-pip -y
pip install -U pip
```

2. install the ultralytics package globally DEPRECATED TO APP SETUP

```bash
pip install ultralytics[export]
```

3. reboot the machine

```bash
sudo reboot
```

4. Install PyTorch and Torchvision  DEPRECATED TO APP SETUP STEP

The above ultralytics installation will install Torch and Torchvision. However, 
these two packages installed via pip are not compatible with the Jetson platform, 
which is based on ARM64 architecture. Therefore, we need to manually install 
a pre-built PyTorch pip wheel and compile or install Torchvision from source.

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```

5. Install cuSPARSELt to fix a dependency issue with torch 2.5.0

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install libcusparselt0 libcusparselt-dev
```

6. Install onnxruntime-gpu

The onnxruntime-gpu package hosted in PyPI does not have aarch64 binaries 
for the Jetson. So we need to manually install this package. This package 
is needed for some of the exports.

```bash
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl
```

## Initializing application

Now it's time to install this repo onto the Jetson and get it running. 

1. If you haven't already done so, clone and enter this repository (user's root 
directory is fine, doesn't really matter)

```bash
cd
git clone https://github.com/topper3418/vision-location-detector.git
cd vision-location-detector
```

2. Activate and set up virtual environment

```bash
python -m venv venv --system-site-packages
# Activate venv
source venv/bin/activate

# Install the Jetson-specific wheels INSIDE the venv (these provide GPU accel)
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torch-2.5.0a0+872d972e41.nv24.08-cp310-cp310-linux_aarch64.whl
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl

# onnxruntime-gpu for exports
pip install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl

# Then ultralytics (with export extras)
pip install -U ultralytics[export]

# also consider installing uv for this
# or try doing it more lightweight, might save time
pip install -U ultralytics  # Base install – quick, no big extras
pip install onnx onnxslim   # Enough for ONNX → TensorRT export on Jetson

# Install other project deps
pip install -r requirements.txt
```

3. copy example.env to .env to use the example configuration

```bash
cp example.env .env
```

4. Run for the first time to test and to initialize the engine that cuda will use

```bash
./run.sh
```

-or-

```bash
python -m src.main
```