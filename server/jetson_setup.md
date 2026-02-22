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

There is a script that installs all required non-venv dependencies. For details,
read the comments in the script

1. Navigate to the project directory

```bash
cd vision-location-detector
```

2. Run the script

```bash
./server/install_external_deps.sh
```

## Initializing application

There is a script that initializes and enters venv, and installs all required
dependencies within it. 

```bash
./scripts/install_internal_deps.sh
```