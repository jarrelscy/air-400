#!/bin/bash
set -e

MODEL_DIR="/app/models_pretrained"

# Download checkpoint if not present
if [ ! -f "$MODEL_DIR/checkpoint/VIRENet_best.pth" ]; then
    echo "Downloading VIRENet checkpoint..."
    mkdir -p "$MODEL_DIR/checkpoint"
    gdown --folder "https://drive.google.com/drive/folders/1ohYbeIJG85cpop3yhBtXaCfQ3ooWZMsk" -O "$MODEL_DIR/checkpoint" || \
        echo "Warning: Could not download checkpoint"
fi

# Download detectors if not present
if [ ! -f "$MODEL_DIR/detectors/yolov8m.pt" ]; then
    echo "Downloading ROI detectors..."
    mkdir -p "$MODEL_DIR/detectors"
    gdown --folder "https://drive.google.com/drive/folders/1PQo7md-hW1x76l_GaBnWH8_H8U7rxpOt" -O "$MODEL_DIR/detectors" || {
        echo "Warning: Could not download detectors from Google Drive, using default YOLO"
        python -c "from ultralytics import YOLO; m = YOLO('yolov8m.pt'); import shutil; shutil.copy(m.ckpt_path, '$MODEL_DIR/detectors/yolov8m.pt')"
    }
fi

echo "Models ready. Starting service..."
exec "$@"
