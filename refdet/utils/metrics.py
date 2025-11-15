"""Metrics for patch-based detection."""

import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert boxes from [x_c, y_c, w, h] to [x1, y1, x2, y2] format."""
    x_c, y_c, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = x_c - w / 2
    y1 = y_c - h / 2
    x2 = x_c + w / 2
    y2 = y_c + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of boxes in [x_c, y_c, w, h] format."""
    # Convert to [x1, y1, x2, y2] format
    boxes1_xyxy = box_cxcywh_to_xyxy(boxes1)
    boxes2_xyxy = box_cxcywh_to_xyxy(boxes2)
    
    # Compute intersection
    x1 = torch.max(boxes1_xyxy[..., 0], boxes2_xyxy[..., 0])
    y1 = torch.max(boxes1_xyxy[..., 1], boxes2_xyxy[..., 1])
    x2 = torch.min(boxes1_xyxy[..., 2], boxes2_xyxy[..., 2])
    y2 = torch.min(boxes1_xyxy[..., 3], boxes2_xyxy[..., 3])

    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    
    # Compute areas
    area1 = (boxes1_xyxy[..., 2] - boxes1_xyxy[..., 0]).clamp(min=0) * (boxes1_xyxy[..., 3] - boxes1_xyxy[..., 1]).clamp(min=0)
    area2 = (boxes2_xyxy[..., 2] - boxes2_xyxy[..., 0]).clamp(min=0) * (boxes2_xyxy[..., 3] - boxes2_xyxy[..., 1]).clamp(min=0)
    
    # Compute union
    union = area1 + area2 - inter + 1e-6
    
    return inter / union


def compute_patch_accuracy(pred_probs: torch.Tensor, target_heatmaps: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute patch classification accuracy.
    
    Args:
        pred_probs: (B, 16, 1) - Predicted probabilities (already sigmoid)
        target_heatmaps: (B, 16) - Target heatmaps (0 or 1)
        threshold: Classification threshold
        
    Returns:
        accuracy: Patch classification accuracy
    """
    # pred_probs already sigmoid activated from model
    pred_probs_flat = pred_probs.squeeze(-1)  # (B, 16)
    pred_binary = (pred_probs_flat > threshold).float()
    
    correct = (pred_binary == target_heatmaps).float()
    accuracy = correct.mean().item()
    
    return accuracy


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.5) -> torch.Tensor:
    """
    Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.
    
    Algorithm:
    1. Sort boxes by confidence scores (descending)
    2. Keep box with highest score
    3. Remove all boxes with IoU > threshold with kept box
    4. Repeat until no boxes left
    
    Args:
        boxes: (N, 4) - Bounding boxes in [x_c, y_c, w, h] format (normalized 0-1)
        scores: (N,) - Confidence scores for each box
        iou_threshold: IoU threshold for suppression (default 0.5)
        
    Returns:
        keep_indices: (M,) - Indices of boxes to keep after NMS
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    # Convert to xyxy format for IoU computation
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)
    
    # Sort boxes by confidence scores (descending)
    sorted_indices = torch.argsort(scores, descending=True)
    
    keep_indices = []
    
    while len(sorted_indices) > 0:
        # Keep box with highest score
        current_idx = sorted_indices[0].item()
        keep_indices.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
        
        # Get current box
        current_box = boxes_xyxy[current_idx:current_idx+1]  # (1, 4)
        
        # Get remaining boxes
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes_xyxy[remaining_indices]  # (M, 4)
        
        # Compute IoU between current box and remaining boxes
        ious = compute_iou(
            boxes[current_idx:current_idx+1],  # (1, 4) in cxcywh
            boxes[remaining_indices]  # (M, 4) in cxcywh
        ).squeeze(0)  # (M,)
        
        # Keep boxes with IoU <= threshold (not overlapping too much)
        mask = ious <= iou_threshold
        sorted_indices = remaining_indices[mask]
    
    return torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)
