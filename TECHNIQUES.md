# Kỹ Thuật Áp Dụng trong Reference-based Detection

Tài liệu này mô tả chi tiết các kỹ thuật đã được áp dụng trong hệ thống Reference-based Detection cho drone surveillance, bao gồm kiến trúc model, loss functions, và các kỹ thuật training.

---

## 📐 1. Kiến Trúc Model

### 1.1 Siamese Network Architecture

**Mô tả:** Sử dụng kiến trúc Siamese với shared backbone để extract features từ cả template và search images.

**Ý tưởng:**
- Template (reference image) và Search (query frame) dùng chung một backbone
- Đảm bảo feature space consistency giữa reference và query
- Hiệu quả về tham số và training stability

**Paper tham khảo:**
- Siamese Neural Networks for One-shot Image Recognition (2015)
- Fully-Convolutional Siamese Networks for Object Tracking (2016)

**Implementation:**
```python
# Shared EfficientNet-B3 backbone
template_feat = backbone(template)  # (B, 512, H, W)
search_feat = backbone(search)      # (B, 512, H, W)
```

---

### 1.2 EfficientNet-B3 Backbone

**Mô tả:** Sử dụng EfficientNet-B3 làm feature extractor với pretrained ImageNet weights.

**Ưu điểm:**
- Compound scaling (depth, width, resolution) (2019)
- Hiệu quả về tham số và FLOPs
- Tốt cho small object detection nhờ multi-scale features

**Paper tham khảo:**
- EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (2019)

**Configuration:**
- Input: 640×640×3
- Output: (B, 512, H, W) sau projection head
- Pretrained: ImageNet weights

---

### 1.3 Patch-based Detection

**Mô tả:** Chia search image thành 4×4 grid (16 patches) để detect object ở từng patch.

**Ý tưởng:**
- Thay vì predict toàn bộ image, predict từng patch
- Mỗi patch có classification score và bbox regression
- Phù hợp với small object detection trong drone surveillance

**Ưu điểm:**
- Tăng resolution cho small objects
- Localize chính xác hơn
- Giảm false positives

**Paper tham khảo:**
- You Only Look Once: Unified, Real-Time Object Detection (2016)
- FCOS: Fully Convolutional One-Stage Object Detection (2019)

**Implementation:**
```python
# Split search features into 4×4 grid
search_patches = patch_embed(search_feat)  # (B, 16, 512)
```

---

### 1.4 Transformer Architecture

#### 1.4.1 Self-Attention Layers

**Mô tả:** Self-attention giữa các search patches để capture spatial relationships.

**Ý tưởng:**
- Mỗi patch attend đến tất cả patches khác
- Học được context và spatial dependencies
- Quan trọng cho việc phân biệt object vs background

**Paper tham khảo:**
- Attention Is All You Need (2017)
- Vision Transformer (ViT) (2020)

**Configuration:**
- Number of layers: 4
- Number of heads: 8
- Embedding dim: 512
- Dropout: 0.1

#### 1.4.2 Cross-Attention

**Mô tả:** Cross-attention giữa search patches (Query) và reference tokens (Key/Value).

**Ý tưởng:**
- Search patches query information từ reference features
- Match template với search patches
- Tăng accuracy cho reference-based detection

**Paper tham khảo:**
- Attention Is All You Need (2017)
- DETR: End-to-End Object Detection with Transformers (2020)

**Implementation:**
```python
# Cross-attention: Search patches attend to reference
attended_patches = cross_attn(search_patches, ref_tokens)
```

---

### 1.5 Detection Heads

**Mô tả:** Hai heads riêng biệt cho classification và regression.

**Architecture:**
- **Classification Head:** MLP → Sigmoid → (B, 16, 1)
- **Regression Head:** MLP → (B, 16, 4) - bbox deltas

**Paper tham khảo:**
- Faster R-CNN: Towards Real-Time Object Detection (2015)
- FCOS: Fully Convolutional One-Stage Object Detection (2019)

---

## 🎯 2. Loss Functions

### 2.1 Focal Loss

**Mô tả:** Focal Loss để xử lý class imbalance (15 negative patches : 1 positive patch).

**Công thức:**
```
FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
```

**Tham số:**
- `α = 0.25`: Weighting factor cho positive class
- `γ = 2.0`: Focusing parameter (focus on hard examples)

**Ưu điểm:**
- Giảm weight của easy negatives
- Focus vào hard examples
- Cải thiện recall cho small objects

**Paper tham khảo:**
- Focal Loss for Dense Object Detection (2017)

**Implementation:**
```python
focal_loss(pred_probs, targets, alpha=0.25, gamma=2.0, smooth=0.05)
```

---

### 2.2 Label Smoothing

**Mô tả:** Label smoothing để tránh overconfidence và tăng robustness.

**Công thức:**
```
target_smooth = target * (1 - smooth) + smooth / 2
```

**Kết quả:**
- Hard label = 1.0 → 0.975 (với smooth=0.05)
- Hard label = 0.0 → 0.025
- Soft label (IoU) giữ được chênh lệch

**Ưu điểm:**
- Tránh overconfident predictions
- Robust với noise trong video
- Tốt cho small object detection

**Paper tham khảo:**
- Rethinking the Inception Architecture for Computer Vision (2016)
- When Does Label Smoothing Help? (2019)

**Integration:**
- Áp dụng trong Focal Loss để smooth targets trước khi compute cross-entropy

---

### 2.3 IoU-based Soft Labels

**Mô tả:** Sử dụng IoU values làm soft targets thay vì hard labels (0/1).

**Ý tưởng:**
- Patch có IoU cao với object → target cao
- Patch có IoU thấp → target thấp
- Phù hợp với patch-based detection

**Công thức:**
```python
# Compute IoU between patch bbox and object bbox
iou = compute_patch_object_iou(patch_bbox, obj_bbox)
heatmap[patch_idx] = iou  # Use IoU as soft target
```

**Ưu điểm:**
- Fine-grained supervision
- Tốt cho small objects (IoU = 0.15 vẫn có signal)
- Tốt cho objects spanning multiple patches

**Paper tham khảo:**
- IoU-aware Single-stage Object Detector for Accurate Localization (2019)
- Soft Labels for Object Detection (2020)

**Implementation:**
```python
# In make_patch_heatmaps()
for patch in patches:
    iou = compute_patch_object_iou(patch_bbox, obj_bbox)
    heatmap[patch_idx] = iou  # Soft target
```

---

### 2.4 Object Size Normalization

**Mô tả:** Weight loss theo số lượng positive patches để balance giữa small và large objects.

**Công thức:**
```python
num_pos_patches = pos_mask.sum(dim=1)  # (B,)
size_weights = 1.0 / sqrt(num_pos_patches + 1e-6)
weighted_loss = (size_weights * loss_per_sample).mean()
```

**Kết quả:**
- Small object (1 patch) → weight = 1.0
- Large object (4 patches) → weight = 0.5
- Medium object (2 patches) → weight = 0.707

**Ưu điểm:**
- Tránh large objects dominate training
- Tăng focus vào small objects
- Balance detection performance

**Paper tham khảo:**
- Focal Loss for Dense Object Detection (2017) - similar idea for class imbalance
- Learning to Balance: Importance Sampling for Object Detection (2019)

**Implementation:**
```python
# Per-sample weighting
cls_loss_per_sample = focal_loss(..., reduction='none')  # (B,)
weighted_cls_loss = (size_weights * cls_loss_per_sample).mean()
```

---

### 2.5 Smooth L1 Loss (Regression)

**Mô tả:** Smooth L1 loss cho bbox regression, chỉ tính trên positive patches.

**Công thức:**
```
smooth_l1(x) = {
    0.5 * x^2  if |x| < 1
    |x| - 0.5  otherwise
}
```

**Ưu điểm:**
- Robust với outliers
- Smooth gradient
- Chỉ tính trên positive patches (efficient)

**Paper tham khảo:**
- Fast R-CNN (2015)
- Faster R-CNN: Towards Real-Time Object Detection (2015)

**Weight:** `reg_weight = 2.0` (classification loss được weight bởi size_weights)

---

## 🔧 3. Training Techniques

### 3.1 Hard Mining

**Mô tả:** Oversample hard samples (small objects, near boundaries, elongated) để tăng focus vào difficult cases.

**Hard Criteria:**
1. **Small objects:** `area < 0.01` (objects < 64px in 640×640)
2. **Near boundaries:** Object center gần patch boundaries
3. **Elongated objects:** `aspect_ratio > 3.0`

**Implementation:**
- Oversample 33% hard samples trong dataset
- `__len__()` returns `len(samples) + len(hard_samples) // 3`
- `__getitem__()` maps indices to hard samples khi cần

**Ưu điểm:**
- Tăng focus vào difficult cases
- Cải thiện recall cho small objects
- Faster convergence

**Paper tham khảo:**
- Training Region-based Object Detectors with Online Hard Example Mining (2016)
- Focal Loss for Dense Object Detection (2017) - hard example mining concept

---

### 3.2 Data Augmentation

#### 3.2.1 Geometric Augmentation

**Mô tả:** Geometric transformations (rotation, flip, affine) áp dụng đồng bộ cho template và search.

**Transformations:**
- **Rotation:** ±5° (reduced from ±10°)
- **Horizontal flip:** 50% probability
- **Vertical flip:** 30% probability
- **Affine:** Translation (±5%), Scale (0.95-1.05), Shear (±3°)

**Bbox Transformation:**
- Transform bbox coordinates theo đúng geometric augmentations
- Sử dụng `transform_bbox()` để convert 4 corners → center-based format

**Ưu điểm:**
- Tăng data diversity
- Robust với camera motion, rotation
- Maintain template-search alignment

**Paper tham khảo:**
- Data Augmentation for Object Detection (2017)
- Learning Data Augmentation Strategies for Object Detection (2019)

#### 3.2.2 Color Augmentation

**Mô tả:** Color jitter (brightness, contrast, saturation) áp dụng đồng bộ.

**Parameters:**
- **Brightness:** ±30% (reduced from ±40%)
- **Contast:** ±20% (reduced from ±30%)
- **Saturation:** ±20% (reduced from ±30%)

**Lưu ý:** Color augmentation áp dụng cùng cho template và search để maintain feature matching.

**Paper tham khảo:**
- ImageNet Classification with Deep Convolutional Neural Networks (2012)
- AutoAugment: Learning Augmentation Strategies from Data (2018)

---

### 3.3 Gradient Clipping

**Mô tả:** Clip gradients để prevent exploding gradients và NaN losses.

**Implementation:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Ưu điểm:**
- Stable training
- Prevent NaN losses
- Đặc biệt quan trọng với FP32 training

**Paper tham khảo:**
- On the difficulty of training Recurrent Neural Networks (2013)
- Deep Residual Learning for Image Recognition (2016)

---

### 3.4 Learning Rate Scheduling

**Mô tả:** Hỗ trợ 3 loại LR schedule: constant, cosine, linear.

**Options:**
1. **Constant:** Fixed learning rate
2. **Cosine:** Cosine annealing với `min_lr`
3. **Linear:** Linear decay với `min_lr`

**Configuration:**
- Default LR: `1e-4`
- Min LR: `lr * 0.01` (hoặc custom `--min_lr`)
- Cosine: `T_max = epochs`, `eta_min = min_lr`

**Paper tham khảo:**
- SGDR: Stochastic Gradient Descent with Warm Restarts (2016)
- Super-Convergence: Very Fast Training of Neural Networks (2017)

---

## 📊 4. Data Processing

### 4.1 Bounding Box Encoding/Decoding

#### 4.1.1 Encoding (Ground Truth → Model Format)

**Mô tả:** Convert normalized bbox (x_c, y_c, w, h) thành patch-relative deltas.

**Process:**
1. Identify positive patches (IoU > 0.3 với object)
2. Compute deltas từ patch center đến object center
3. Normalize deltas by patch size

**IoU-based Assignment:**
```python
# Patch is positive if IoU > threshold (default 0.3)
for patch in patches:
    iou = compute_patch_object_iou(patch_bbox, obj_bbox)
    if iou > 0.3:
        patch_pos_mask[patch_idx] = 1
        patch_deltas[patch_idx] = compute_deltas(...)
```

**Paper tham khảo:**
- You Only Look Once: Unified, Real-Time Object Detection (2016)
- FCOS: Fully Convolutional One-Stage Object Detection (2019)

#### 4.1.2 Decoding (Model Output → Bbox)

**Mô tả:** Convert patch deltas về normalized bbox coordinates.

**Process:**
1. Get best patch (highest classification score)
2. Decode bbox từ patch center + deltas
3. Clamp to [0, 1] range

**Implementation:**
```python
best_patch_idx = cls_probs.argmax(dim=1)
bbox = decode_patch_bbox(patch_idx, deltas, patch_grid_info)
```

---

### 4.2 Bbox Transformation for Augmentation

**Mô tả:** Transform bbox coordinates khi apply geometric augmentation.

**Method:**
1. Convert center-based (x_c, y_c, w, h) → 4 corners
2. Apply transformations (rotation, flip, affine)
3. Convert back to center-based format

**Transformations:**
- Rotation: around image center
- Flip: mirror coordinates
- Affine: translate → rotate → scale → shear

**Paper tham khảo:**
- Data Augmentation for Object Detection (2017)
- Learning Data Augmentation Strategies for Object Detection (2019)

---

## 🎓 5. Tổng Kết

### 5.1 Kỹ Thuật Chính

| Kỹ Thuật | Mục Đích | Paper |
|----------|----------|-------|
| Siamese Network | Feature consistency | Siamese Networks (2015) |
| EfficientNet-B3 | Efficient backbone | EfficientNet (2019) |
| Patch-based Detection | Small object detection | YOLO (2016), FCOS (2019) |
| Transformer (Self/Cross-Attn) | Spatial relationships | Attention Is All You Need (2017) |
| Focal Loss | Class imbalance | Focal Loss (2017) |
| Label Smoothing | Robustness | Inception v3 (2016) |
| IoU Soft Labels | Fine-grained supervision | IoU-aware Detection (2019) |
| Size Normalization | Balance small/large objects | Focal Loss (2017) |
| Hard Mining | Focus on difficult cases | OHEM (2016) |
| Gradient Clipping | Training stability | ResNet (2016) |

### 5.2 Expected Improvements

- **Small Objects:** Recall tăng 15-25% (soft labels + size weighting)
- **Hard Negatives:** Precision tăng 10-15% (focal loss)
- **Convergence:** Nhanh hơn 2-3x (hard mining)
- **Robustness:** Tốt hơn với video noise (label smoothing)

---

## 📚 6. References

### Papers

1. **Siamese Networks:**
   - Siamese Neural Networks for One-shot Image Recognition (2015)
   - Fully-Convolutional Siamese Networks for Object Tracking (2016)

2. **EfficientNet:**
   - EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (2019)

3. **Object Detection:**
   - You Only Look Once: Unified, Real-Time Object Detection (2016)
   - Faster R-CNN: Towards Real-Time Object Detection (2015)
   - FCOS: Fully Convolutional One-Stage Object Detection (2019)
   - IoU-aware Single-stage Object Detector for Accurate Localization (2019)

4. **Transformers:**
   - Attention Is All You Need (2017)
   - Vision Transformer (ViT) (2020)
   - DETR: End-to-End Object Detection with Transformers (2020)

5. **Loss Functions:**
   - Focal Loss for Dense Object Detection (2017)
   - Rethinking the Inception Architecture for Computer Vision (2016)
   - When Does Label Smoothing Help? (2019)

6. **Training Techniques:**
   - Training Region-based Object Detectors with Online Hard Example Mining (2016)
   - Deep Residual Learning for Image Recognition (2016)
   - SGDR: Stochastic Gradient Descent with Warm Restarts (2016)

7. **Data Augmentation:**
   - ImageNet Classification with Deep Convolutional Neural Networks (2012)
   - Data Augmentation for Object Detection (2017)
   - Learning Data Augmentation Strategies for Object Detection (2019)
   - AutoAugment: Learning Augmentation Strategies from Data (2018)

---

**Last Updated:** 2024

