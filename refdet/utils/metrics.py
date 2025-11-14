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
