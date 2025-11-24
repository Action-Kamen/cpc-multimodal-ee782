#!/usr/bin/env bash
set -e

# Go to project root
cd /home/anirudh/wd/rough

# Activate venv
source venv/bin/activate

mkdir -p logs

echo "=== Running IMAGE (real CIFAR-10) CPC ==="
python cpc_project.py \
  --modality image \
  --use-mock-data false \
  --image-root /home/anirudh/wd/rough/data/images \
  --out-dir /home/anirudh/wd/rough/outputs_image \
  --epochs 20 \
  > logs/image.log 2>&1

echo "=== Running AUDIO (mock sine waves) CPC ==="
python cpc_project.py \
  --modality audio \
  --use-mock-data true \
  --out-dir /home/anirudh/wd/rough/outputs_audio \
  --epochs 10 \
  > logs/audio.log 2>&1

echo "=== Running MULTIMODAL (mock audio+image) CPC ==="
python cpc_project.py \
  --modality multimodal \
  --use-mock-data true \
  --out-dir /home/anirudh/wd/rough/outputs_multimodal \
  --epochs 10 \
  > logs/multimodal.log 2>&1

echo "=== All runs finished ==="
