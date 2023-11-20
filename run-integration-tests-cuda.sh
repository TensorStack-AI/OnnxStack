#! /bin/bash
# running this requires:
# - nvidia GPU with sufficient VRAM
# - nvidia drivers installed on the host system
# - nvidia-container-toolkit installed on the host system (see: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
# - nvidia-smi also reports peak VRAM close 24GB while running the tests
docker-compose up --build