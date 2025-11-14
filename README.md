# RefDet - Phát hiện đối tượng dựa trên ảnh tham chiếu

## 📋 Giới thiệu

RefDet là pipeline phát hiện vật thể nhỏ trong video drone bằng cách so khớp ảnh tham chiếu (template) với ảnh tìm kiếm (search). Toàn bộ mô tả dưới đây sử dụng tiếng Việt để dễ vận hành và chia sẻ nội bộ.

## 🏗️ Kiến trúc Model V2

- **Backbone**: EfficientNet-B3 (tiền huấn luyện ImageNet) chia sẻ cho template và search.
- **Kích thước đầu vào**: 640×640, 3 kênh.
- **Số kênh đặc trưng**: 512D sau projection head.
- **Lưới patch**: 4×4 = 16 patch cho ảnh search.
- **Cơ chế Attention**:
  - Self-Attention trên 16 patch của ảnh search.
  - Cross-Attention: patch ảnh search (Query) tham chiếu tokens không gian của template (Key/Value).
- **Tham số**: 31.71M (~121 MB FP32 / 61 MB FP16).
- **Đầu ra**: 16 xác suất (cls) + 16 bbox delta (reg) theo patch.

### Sơ đồ luồng xử lý

1. **Template**  
   - Ảnh template → EfficientNet-B3 → Tensor 512×H×W.  
   - AdaptiveAvgPool2d(4×4) → Flatten → 16 tokens (mỗi token 512D) → LayerNorm.
2. **Search**  
   - Ảnh search → EfficientNet-B3 → Tensor 512×H×W.  
   - Chia 4×4 patch, Flatten từng patch → Linear thích nghi → Cộng positional embedding 2D.  
   - Qua `num_layers` TransformerBlock (self-attention + FFN).
3. **Cross-Attention**  
   - Query: patch search sau self-attention.  
   - Key/Value: 16 tokens của template.  
   - Kết quả được chuẩn hóa (LayerNorm).
4. **Đầu dự đoán**  
   - Linear đưa patch trở lại bố cục 4×4 → Conv refinement.  
   - **Cls head**: Flatten → MLP → Sigmoid → xác suất patch.  
   - **Reg head**: Flatten → MLP → 16 × (dx, dy, dw, dh).

### Ưu điểm chính
- Template và search chia sẻ backbone → giảm tham số.
- Patch grid 4×4 + positional embedding giúp bắt vật thể nhỏ và giữ thông tin vị trí.
- Cross-attention trực tiếp giữa patch search và tokens không gian của template → tăng độ chính xác truy hồi.
- Head conv + MLP giúp tinh chỉnh đặc trưng không gian trước khi dự đoán.

## 📁 Cấu trúc thư mục

```
refdet/
├── data_process/              # Script xử lý dữ liệu
│   ├── prepare_retrieval_dataset_flat.py   # Tách frame + tạo label YOLO
│   ├── create_zoomed_dataset.py            # Nhân đôi data bằng zoom
│   └── fix_labels.py                       # Đảm bảo mỗi file chỉ 1 bbox
├── refdet/                   # Source code model V2
│   ├── model.py              # Định nghĩa kiến trúc
│   ├── train.py              # Vòng huấn luyện + đánh giá
│   └── utils/                # Dataset, geometry, metrics, transforms
├── retrieval_dataset_flat/          # Dataset gốc (sau khi chuẩn hóa)
├── retrieval_dataset_flat_zoomed/   # Dataset gốc + bản zoom (x2 size)
└── requirements.txt                # Thư viện cần cài
```

## ⚙️ Chuẩn bị môi trường

```bash
conda activate zlai
pip install -r requirements.txt
```

## 🗂️ Xử lý dữ liệu

1. **Trích xuất frame + template**  
   ```bash
   cd refdet/data_process
   python prepare_retrieval_dataset_flat.py \
       --source_dir ../train \
       --output_dir ../retrieval_dataset_flat
   ```
2. **Fix label (mỗi file 1 bbox)**  
   ```bash
   python fix_labels.py --data_dir ../retrieval_dataset_flat
   ```
3. **Nhân đôi data bằng zoom**  
   ```bash
   python create_zoomed_dataset.py \
       --source_dir ../retrieval_dataset_flat \
       --output_dir ../retrieval_dataset_flat_zoomed \
       --area_ratio1 0.15 \
       --area_ratio2 0.35 \
       --area_ratio3 0.55 \
       --area_ratio4 0.75
   ```

## 🧠 Nhu cầu VRAM & Training

- **Inference (batch=1)**: ≈ 1–2 GB.
- **Training FP32 (batch=8)**: ≈ 12–15 GB.
- **Training FP32 (batch=16)**: ≈ 23–24 GB (đã đo trên RTX 3090).
- **Training FP16 (batch=8)**: ≈ 6–8 GB.
- **Khuyến nghị**: GPU ≥ 8 GB, ưu tiên FP16 + batch 8 để ổn định.

### Chạy huấn luyện

```bash
cd refdet/refdet
python train.py \
    --data_dir ../retrieval_dataset_flat_zoomed \
    --batch_size 8 \
    --epochs 30 \
    --lr 1e-4 \
    --num_heads 8 \
    --num_layers 2 \
    --dropout 0.1 \
    --workers 4
```

### Siêu tham số quan trọng

- `--data_dir`: thư mục dataset (nên trỏ tới bản zoomed).
- `--batch_size`: mặc định 16, giảm xuống 8 nếu GPU 8 GB.
- `--num_heads`, `--num_layers`: điều chỉnh độ rộng/ sâu của attention stack.
- `--dropout`: 0.1 giúp regularize patch features.
- `--workers`: số tiến trình load dữ liệu (4 là an toàn).

### Checkpoint

- `checkpoints/best_model_rank{1..3}.pt`: lưu theo mIoU cao nhất.
- `checkpoints/last_model_epoch_N.pt`: epoch cuối cùng.
- `checkpoints/checkpoint_epoch_N.pt`: lưu chu kỳ 20 epoch (model + optimizer + scaler).

## 📊 Chỉ số theo dõi

- **mIoU**: trung bình IoU sau khi decode bbox.
- **Patch Accuracy**: tỉ lệ patch được phân loại đúng (có/không có vật thể).

## 📦 Cấu trúc dataset

```
retrieval_dataset_flat/
├── train/
│   ├── templates/          # Ảnh tham chiếu (copy cho cả train/val)
│   └── search/
│       ├── images/         # Frame đã trích
│       └── labels/         # Nhãn YOLO (class x_c y_c w h)
└── val/
    ├── templates/
    └── search/
        ├── images/
        └── labels/
```

Dataset `retrieval_dataset_flat_zoomed` có cùng cấu trúc nhưng số lượng ảnh gấp đôi (ảnh gốc + ảnh zoom theo tỷ lệ 15/35/55/75% diện tích).

## 🛠️ Công cụ hỗ trợ

- **Sửa nhãn**  
  ```bash
  python data_process/fix_labels.py --data_dir retrieval_dataset_flat
  ```
- **Tạo dataset zoom**  
  ```bash
  python data_process/create_zoomed_dataset.py \
      --source_dir retrieval_dataset_flat \
      --output_dir retrieval_dataset_flat_zoomed \
      --area_ratio1 0.15 --area_ratio2 0.35 \
      --area_ratio3 0.55 --area_ratio4 0.75
  ```

## 📝 Ghi chú

- **Tham số model**: 31.71M (< 50M theo yêu cầu).
- **Dataset zoomed**: nên dùng cho training để cải thiện recall.
- **Định dạng nhãn**: YOLO chuẩn `class_id x_c y_c w h` (0–1).
- **Mixed Precision (AMP)**: bật mặc định trong `train.py`, giúp tiết kiệm ~40% VRAM.
- **Batch size**: luôn điều chỉnh theo dung lượng VRAM thực tế; giảm batch trước khi giảm kiến trúc.
