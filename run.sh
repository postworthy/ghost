#!/bin/bash
cd "$(dirname "$0")"

docker build -f Dockerfile . -t ghost-onnx-export:latest
docker run -it --shm-size=2gb --gpus all -v /mnt/d/TrainingData/vggface2/:/VggFace2-crop -v ./output:/app/ghost/output/  ghost-onnx-export:latest