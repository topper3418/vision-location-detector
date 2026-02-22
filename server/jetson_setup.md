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

3. Insert the SD card to the Jetson, power it on, and follow the onscreen 
prompts to set up the machine (you'll need a monitor, keyboard and mouse)

4. Update all software packages and reboot

```bash
sudo apt update && sudo apt upgrade -y && sudo reboot
```

## Installing dependencies

Not all default pip repos are configured to properly utilize the jetson's
GPU to its fullest potential, thus the dependency install process is quite 
complicated. Thus, scripts have been written to simplify the process and 
ensure consistent installs

1. Navigate to the project directory

```bash
cd vision-location-detector
```

2. Run the external dependency script

```bash
./server/install_external_deps.sh
```

3. run the internal dependency script

```bash
./scripts/install_internal_deps.sh
```