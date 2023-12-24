#!/bin/bash
cd "$(dirname "$0")"

docker build -f Dockerfile . -t ghost-onnx-export:latest
docker run -it --gpus all -v ./output:/app/ghost/output/ ghost-onnx-export:latest