#!/bin/bash
# Download pretrained models for baby monitor service
# Run this script before starting the container

set -e

MODEL_DIR="${1:-/data/baby-monitor/models}"

echo "Creating model directories..."
mkdir -p "$MODEL_DIR/checkpoint"
mkdir -p "$MODEL_DIR/detectors"

echo "Installing gdown..."
pip install gdown --quiet

echo "Downloading VIRENet checkpoint..."
cd "$MODEL_DIR/checkpoint"
gdown --folder https://drive.google.com/drive/folders/1ohYbeIJG85cpop3yhBtXaCfQ3ooWZMsk --quiet || {
    echo "Failed to download checkpoint. Please download manually from:"
    echo "https://drive.google.com/drive/folders/1ohYbeIJG85cpop3yhBtXaCfQ3ooWZMsk"
}

echo "Downloading ROI detectors..."
cd "$MODEL_DIR/detectors"
gdown --folder https://drive.google.com/drive/folders/1PQo7md-hW1x76l_GaBnWH8_H8U7rxpOt --quiet || {
    echo "Failed to download detectors. Please download manually from:"
    echo "https://drive.google.com/drive/folders/1PQo7md-hW1x76l_GaBnWH8_H8U7rxpOt"
}

echo ""
echo "Models downloaded to: $MODEL_DIR"
echo ""
echo "Expected structure:"
echo "$MODEL_DIR/"
echo "├── checkpoint/"
echo "│   └── VIRENet_best.pth"
echo "└── detectors/"
echo "    ├── yolov8m.pt"
echo "    └── yolov8n-face.pt"
echo ""
echo "Make sure compose.yaml volume mount points to: $MODEL_DIR:/app/models_pretrained"
