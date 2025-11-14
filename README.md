# RefDet - Phát hiện đối tượng dựa trên ảnh tham chiếu

## 📋 Giới thiệu

RefDet phát hiện vật thể nhỏ trong video drone bằng cách so khớp ảnh tham chiếu (template) với ảnh tìm kiếm (search).

**Kiến trúc**: EfficientNet-B3 backbone + Transformer (4 layers, 8 heads) + Patch-based detection (4×4 grid)

## ⚙️ Cài đặt

```bash
pip install -r requirements.txt
```

## 🚀 Training

### Training từ đầu

```bash
cd refdet
python train.py \
    --data_dir /path/to/dataset \
    --output_dir outputs_v2 \
    --batch_size 8 \
    --epochs 60 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --augment_prob 0.4 \
    --num_heads 8 \
    --num_layers 4 \
    --dropout 0.1 \
    --workers 4
```

### Resume từ checkpoint

```bash
cd refdet
python train.py \
    --data_dir /path/to/dataset \
    --output_dir outputs_v2 \
    --checkpoint_path outputs_v2/checkpoint_epoch_2.pth \
    --batch_size 60 \
    --epochs 12 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --augment_prob 0.4 \
    --num_heads 8 \
    --num_layers 4 \
    --dropout 0.1 \
    --workers 4
```

**Lưu ý**: Khi resume, các tham số model (`num_heads`, `num_layers`, `dropout`) phải khớp với checkpoint. Các tham số training (`lr`, `batch_size`, `augment_prob`) có thể thay đổi.

### Tham số quan trọng

- `--augment_prob`: Xác suất augment (mặc định 0.2 = 20%)
- `--checkpoint_path`: Đường dẫn checkpoint để resume training
- `--batch_size`: Mặc định 16, giảm xuống 8 nếu GPU < 16GB
- `--epochs`: Số epoch (mặc định 80)

## 🔍 Inference

```bash
cd refdet
python inference.py \
    --checkpoint_path outputs_v2/best_model_epoch_X_mIoU_X.XXXX.pth \
    --data_dir /path/to/dataset \
    --split public_test \
    --output_dir ./results \
    --confidence_threshold 0.5
```

### Tham số inference

- `--checkpoint_path`: Đường dẫn model checkpoint (bắt buộc)
- `--data_dir`: Thư mục dataset root (bắt buộc)
- `--split`: Dataset split (mặc định: `public_test`)
- `--output_dir`: Thư mục output - sẽ lưu `submission.json` trong thư mục này (bắt buộc)
- `--confidence_threshold`: Ngưỡng confidence (mặc định: 0.5)

## 📁 Cấu trúc Dataset

```
dataset/
├── train/
│   ├── templates/          # Ảnh tham chiếu
│   └── search/
│       ├── images/         # Frame đã trích
│       └── labels/         # Nhãn YOLO (class x_c y_c w h)
├── val/
│   └── ... (tương tự)
└── public_test/
    ├── templates/
    └── search/
        └── images/
```

## 💾 Checkpoint

- `best_model_epoch_X_mIoU_X.XXXX.pth`: Model tốt nhất (theo mIoU)
- `last_model_epoch_X.pth`: Model epoch cuối
- `checkpoint_epoch_X.pth`: Checkpoint đầy đủ (model + optimizer + scaler) - lưu mỗi 20 epochs

## 📊 Metrics

- **mIoU**: Mean IoU sau khi decode bbox
- **Patch Accuracy**: Tỉ lệ patch được phân loại đúng
- **Loss**: Classification loss + Regression loss

## 🧠 Yêu cầu VRAM

- **Inference**: ~1-2 GB
- **Training (batch=8)**: ~6-8 GB (FP16) / ~12-15 GB (FP32)
- **Training (batch=16)**: ~23-24 GB (FP32)

**Khuyến nghị**: GPU ≥ 8GB, dùng FP16 + batch_size=8

## 🗂️ Xử lý dữ liệu

### 1. Trích xuất frame + template

```bash
cd data_process
python prepare_retrieval_dataset_flat.py \
    --source_dir ../train \
    --output_dir ../retrieval_dataset_flat
```

### 2. Fix label (mỗi file 1 bbox)

```bash
python fix_labels.py --data_dir ../retrieval_dataset_flat
```

### 3. Tạo dataset zoom (tùy chọn)

```bash
python create_zoomed_dataset.py \
    --source_dir ../retrieval_dataset_flat \
    --output_dir ../retrieval_dataset_flat_zoomed \
    --area_ratio1 0.15 \
    --area_ratio2 0.35 \
    --area_ratio3 0.55 \
    --area_ratio4 0.75
```

## 📝 Ghi chú

- Model sử dụng **Mixed Precision (AMP)** tự động để tiết kiệm VRAM
- **Augment probability**: 20% để giữ phân phối dữ liệu gốc
- **Output format**: submission.json theo format yêu cầu với `video_id`, `detections`, `bboxes` (frame, x1, y1, x2, y2)

## 🏗️ Kiến trúc Model

### Sơ đồ tổng quan

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Enhanced Siamese Detector V2                     │
└─────────────────────────────────────────────────────────────────────────┘

INPUT: Template (640×640×3)          INPUT: Search (640×640×3)
         │                                    │
         ▼                                    ▼
┌────────────────────┐            ┌────────────────────┐
│  EfficientNet-B3   │            │  EfficientNet-B3   │
│   (Shared)         │            │   (Shared)         │
└────────────────────┘            └────────────────────┘
         │                                    │
    (B,512,H,W)                          (B,512,H,W)
         │                                    │
         ▼                                    ▼
┌────────────────────┐            ┌────────────────────┐
│ AdaptiveAvgPool2d  │            │   PatchEmbedding   │
│      (4×4)         │            │    (4×4 grid)      │
└────────────────────┘            └────────────────────┘
         │                                    │
    (B,512,4,4)                          (B,16,512)
         │                                    │
         ▼                                    ▼
┌────────────────────┐            ┌────────────────────┐
│   Flatten + Permute│            │  Self-Attention     │
│   + LayerNorm      │            │  (4 layers)         │
└────────────────────┘            └────────────────────┘
         │                                    │
    (B,16,512)                           (B,16,512)
         │                                    │
         └──────────────┬─────────────────────┘
                        │
                        ▼
              ┌────────────────────┐
              │  Cross-Attention   │
              │  Q: Search patches │
              │  K,V: Ref tokens    │
              └────────────────────┘
                        │
                   (B,16,512)
                        │
                        ▼
              ┌────────────────────┐
              │   Final LayerNorm  │
              └────────────────────┘
                        │
                        ▼
              ┌────────────────────┐
              │  Spatial Refine    │
              │  Linear+Norm+ReLU  │
              └────────────────────┘
                        │
                   (B,16,512)
                        │
                        ▼
              ┌────────────────────┐
              │  Reshape to 4×4   │
              │  (B,512,4,4)       │
              └────────────────────┘
                        │
                        ▼
              ┌────────────────────┐
              │  Conv Refinement   │
              │  2× Conv2d+BN+ReLU│
              └────────────────────┘
                        │
                   (B,256,4,4)
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
┌─────────────────┐          ┌─────────────────┐
│  Cls Head       │          │  Reg Head        │
│  Flatten        │          │  Flatten         │
│  MLP(256→128→64)│          │  MLP(256→128→64) │
│  Sigmoid        │          │  Linear(64→64)   │
└─────────────────┘          └─────────────────┘
         │                             │
    (B,16,1)                      (B,16,4)
         │                             │
         └──────────────┬──────────────┘
                        │
                        ▼
              ┌────────────────────┐
              │   Output           │
              │   - cls_probs       │
              │   - bbox_deltas     │
              └────────────────────┘
```

### Chi tiết các thành phần

#### 1. Backbone (EfficientNet-B3)
- **Input**: 640×640×3
- **Output**: (B, 512, H, W) sau projection head
- **Shared**: Template và Search dùng chung backbone

#### 2. Template Processing
```
Template (640×640×3)
  → EfficientNet-B3 → (B, 512, H, W)
  → AdaptiveAvgPool2d(4×4) → (B, 512, 4, 4)
  → Flatten(2) + Permute(0,2,1) → (B, 16, 512)
  → LayerNorm → (B, 16, 512) [Reference Tokens]
```

#### 3. Search Processing
```
Search (640×640×3)
  → EfficientNet-B3 → (B, 512, H, W)
  → PatchEmbedding (4×4 grid) → (B, 16, 512)
  → Self-Attention (4 layers) → (B, 16, 512)
```

#### 4. Cross-Attention
- **Query**: Search patches (B, 16, 512)
- **Key/Value**: Reference tokens (B, 16, 512)
- **Output**: Attended patches (B, 16, 512)

#### 5. Detection Heads
```
Attended patches (B, 16, 512)
  → Spatial Refine (Linear+Norm+ReLU) → (B, 16, 512)
  → Reshape to 4×4 → (B, 512, 4, 4)
  → Conv Refinement (2× Conv2d) → (B, 256, 4, 4)
  → Classification Head: Flatten → MLP → Sigmoid → (B, 16, 1)
  → Regression Head: Flatten → MLP → (B, 16, 4)
```

### Tham số Model

- **Backbone**: EfficientNet-B3 (pretrained ImageNet)
- **Embedding dim**: 512
- **Patch grid**: 4×4 = 16 patches
- **Transformer layers**: 4 (self-attention)
- **Attention heads**: 8
- **Dropout**: 0.1
- **Total params**: ~31.71M
