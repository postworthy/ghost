#!/bin/bash
cd "$(dirname "$0")"

docker build -f Dockerfile . -t ghost-onnx-export:latest
docker run -it --shm-size=2gb --gpus all -v /mnt/d/TrainingData/img_align_celeba/img_align_celeba_crop/:/img_align_celeba_crop -v /mnt/d/TrainingData/img_align_celeba/img_align_celeba/:/img_align_celeba -v /mnt/d/TrainingData/vggface2-crop/:/VggFace2-crop -v /mnt/d/TrainingData/vggface2/:/VggFace2 -v ./output:/app/ghost/output/  ghost-onnx-export:latest