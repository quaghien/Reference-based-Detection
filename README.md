# Siamese Retrieval Detector for Small Object Detection in Drone Clips

## Overview

A lightweight Siamese network-based object detector designed for finding small objects in drone video frames by matching against reference template images. The system uses a shared backbone encoder with depthwise cross-correlation to generate 2D similarity heatmaps, followed by regression heads for precise bounding box localization.

<img src="images/refdet.png" alt="Architecture Overview" width="70%">

## Key Technical Features

### Architecture
- **Siamese Backbone**: Shared feature encoder (MobileNetV3-Large, EfficientNet-B0, or ResNet18) pretrained on ImageNet
- **Depthwise Cross-Correlation**: Efficient template-search feature matching inspired by SiamFC/SiamRPN
- **Dual-Head Design**: 
  - Classification head outputs 2D heatmap (32×32 grid) for object presence
  - Regression head predicts bounding box offsets (dx, dy, dw, dh) in log-space
- **Heatmap-based Detection**: Gaussian heatmap encoding for center localization with scale-adaptive bbox regression

### Small Object Detection Challenges
- Objects in drone footage are typically small relative to frame size
- High-altitude perspective causes scale variations
- Heatmap-based approach with multi-scale feature matching handles small objects effectively
- Input resolution: 320×320 (configurable) with 32×32 heatmap output (10× downsampling)

### Hardware Constraints
- **Target Device**: NVIDIA Jetson Xavier NX (16GB RAM, Volta GPU)
- **Model Size**: <50M parameters (FP16/INT8 quantization supported)
- **Environment**: CUDA 11.4, cuDNN 8.4, TensorRT 8.4, PyTorch 1.12.1
- **Deployment**: Offline inference, no cloud dependencies

## Project Structure

```
refdet/
├── model.py              # SiameseRetrievalDetector architecture
├── train.py              # Training script with BCE + SmoothL1 loss
├── inference.py          # Inference with pre-encoded template features
├── data_process/
│   └── prepare_retrieval_dataset_flat.py  # Extract frames & prepare dataset
└── utils/
    ├── dataset.py        # RetrievalDataset with heatmap/bbox encoding
    ├── geometry.py       # Heatmap generation & bbox delta encoding
    ├── postprocess.py    # Heatmap → bbox decoding
    ├── transforms.py     # Image preprocessing & augmentation
    └── metrics.py        # IoU computation
```

## Training Pipeline

1. **Dataset Preparation**: Extract annotated frames from drone videos, pair with reference templates
2. **Heatmap Encoding**: Generate Gaussian heatmaps centered at ground-truth object locations
3. **Bbox Encoding**: Encode bounding box deltas relative to heatmap grid positions
4. **Loss Functions**: 
   - Classification: Binary Cross-Entropy on heatmap logits
   - Regression: SmoothL1 on bbox deltas (masked at positive locations)

## Inference Pipeline

1. **Template Pre-encoding**: Extract features from reference images once (cached in GPU)
2. **Frame Processing**: Encode search frame features
3. **Cross-Correlation**: Match template features with frame features
4. **Peak Detection**: Find maximum heatmap response
5. **Bbox Decoding**: Regress bounding box from peak location using predicted deltas

## Usage

### Prepare Dataset
```bash
python data_process/prepare_retrieval_dataset_flat.py
```

### Train Model
```bash
python train.py \
  --data_dir retrieval_dataset_flat \
  --backbone mobilenet_v3_large \
  --channels 256 \
  --img_size 320 \
  --heatmap_size 32 \
  --batch_size 16 \
  --epochs 30 \
  --lr 1e-3
```

### Run Inference
```bash
python inference.py \
  --checkpoint outputs/best_model.pth \
  --template_dir retrieval_dataset_flat/val/templates \
  --frame_dir sample_frames \
  --output_dir runs/inference
```

## Technical Specifications

- **Input**: Template image (320×320) + Search frame (320×320)
- **Output**: 2D heatmap (32×32) + Bbox deltas (32×32×4)
- **Feature Channels**: 256 (after backbone + 1×1 conv projection)
- **VRAM Usage**: ~5-6GB (batch_size=16, FP16)
- **Inference Speed**: Template pre-encoding reduces runtime by ~50%

## References

- SiamFC: Fully-Convolutional Siamese Networks for Object Tracking
- SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks
- Designed for competition constraints: offline, open-source, Jetson-compatible

