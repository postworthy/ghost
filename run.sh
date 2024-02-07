#!/bin/bash
cd "$(dirname "$0")"

docker build -f Dockerfile . -t ghost-onnx-export:latest
docker run -it --shm-size=2gb --gpus all \
    -v /mnt/d/TrainingData/sdxl_turbo_faces/:/sdxl_turbo_faces \
    -v /mnt/d/TrainingData/lfw_funneled/ALL:/lfw_funneled \
    -v /mnt/d/TrainingData/lfw_funneled/CROP:/lfw_funneled_crop \
    -v /mnt/d/TrainingData/img_align_celeba/img_align_celeba_crop/:/img_align_celeba_crop \
    -v /mnt/d/TrainingData/img_align_celeba/img_align_celeba/:/img_align_celeba \
    -v /mnt/d/TrainingData/vggface2-crop/:/VggFace2-crop \
    -v /mnt/d/TrainingData/vggface2/:/VggFace2 \
    -v /mnt/d/TrainingData/celeb_turbo/:/celeb_turbo \
    -v /mnt/d/TrainingData/DigiFace1M/:/digiface \
    -v /mnt/d/TrainingData/FromBadges/raw:/frombadges \
    -v /mnt/d/TrainingData/FromBadges/crop:/frombadges_crop \
    -v /mnt/d/TrainingData/real_faces_128:/real_faces_128 \
    -v ./output:/app/ghost/output/ \
    ghost-onnx-export:latest