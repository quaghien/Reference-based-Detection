"""Geometry utilities for patch-based detection."""

import torch
import torch.nn.functional as F


def make_patch_heatmaps(center_xy, patch_grid_info: dict, box_wh=None) -> torch.Tensor:
    """
    Create soft heatmaps for patch-based detection using IoU values.
    
    Args:
        center_xy: (x_c, y_c) in normalized coordinates (0-1)
        patch_grid_info: Dict with grid_h, grid_w, num_patches
        box_wh: (w, h) in normalized coordinates for IoU-based soft labels
        
    Returns:
        heatmaps: (num_patches,) - IoU-based soft heatmaps for each patch
    """
    x_c, y_c = center_xy
    grid_h = patch_grid_info['grid_h']  # 4
    grid_w = patch_grid_info['grid_w']  # 4
    num_patches = patch_grid_info['num_patches']  # 16
    
    if box_wh is None:
        # Fallback to old behavior (hard labels)
        patch_y = int(y_c * grid_h)
        patch_x = int(x_c * grid_w)
        patch_y = max(0, min(patch_y, grid_h - 1))
        patch_x = max(0, min(patch_x, grid_w - 1))
        
        heatmaps = torch.zeros(num_patches, dtype=torch.float32)
        target_patch_idx = patch_y * grid_w + patch_x
        heatmaps[target_patch_idx] = 1.0
        return heatmaps
    
    # Soft labels using IoU
    w, h = box_wh
    obj_x1, obj_y1 = x_c - w / 2, y_c - h / 2
    obj_x2, obj_y2 = x_c + w / 2, y_c + h / 2
    obj_bbox = (obj_x1, obj_y1, obj_x2, obj_y2)
    
    heatmaps = torch.zeros(num_patches, dtype=torch.float32)
    
    for py in range(grid_h):
        for px in range(grid_w):
            # Patch bbox coordinates
            patch_x1 = px / grid_w
            patch_y1 = py / grid_h
            patch_x2 = (px + 1) / grid_w
            patch_y2 = (py + 1) / grid_h
            patch_bbox = (patch_x1, patch_y1, patch_x2, patch_y2)
            
            # Compute IoU and use as soft target
            iou = compute_patch_object_iou(patch_bbox, obj_bbox)
            patch_idx = py * grid_w + px
            heatmaps[patch_idx] = iou
    
    return heatmaps


def compute_patch_object_iou(patch_bbox, obj_bbox):
    """
    Compute IoU between patch and object bounding boxes.
    
    Args:
        patch_bbox: (x1, y1, x2, y2) - patch coordinates
        obj_bbox: (x1, y1, x2, y2) - object coordinates
        
    Returns:
        iou: IoU value between 0 and 1
    """
    # Intersection coordinates
    x1 = max(patch_bbox[0], obj_bbox[0])
    y1 = max(patch_bbox[1], obj_bbox[1])
    x2 = min(patch_bbox[2], obj_bbox[2])
    y2 = min(patch_bbox[3], obj_bbox[3])
    
    # Check if there's intersection
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    # Intersection area
    intersection = (x2 - x1) * (y2 - y1)
    
    # Areas
    patch_area = (patch_bbox[2] - patch_bbox[0]) * (patch_bbox[3] - patch_bbox[1])
    obj_area = (obj_bbox[2] - obj_bbox[0]) * (obj_bbox[3] - obj_bbox[1])
    
    # Union area
    union = patch_area + obj_area - intersection
    
    return intersection / (union + 1e-6)


def encode_patch_bbox_deltas(box, patch_grid_info: dict, iou_threshold: float = 0.3):
    """
    Encode bbox deltas for patch-based detection using IoU-based assignment.
    
    Args:
        box: (x_c, y_c, w, h) in normalized coordinates (0-1)
        patch_grid_info: Dict with grid_h, grid_w, num_patches
        iou_threshold: IoU threshold for positive patch assignment
        
    Returns:
        bbox_deltas: (num_patches, 4) - Bbox deltas for each patch
        pos_mask: (num_patches,) - Positive mask
    """
    x_c, y_c, w, h = box
    grid_h = patch_grid_info['grid_h']  # 4
    grid_w = patch_grid_info['grid_w']  # 4
    num_patches = patch_grid_info['num_patches']  # 16
    
    # Object bbox in (x1, y1, x2, y2) format
    obj_x1, obj_y1 = x_c - w / 2, y_c - h / 2
    obj_x2, obj_y2 = x_c + w / 2, y_c + h / 2
    obj_bbox = (obj_x1, obj_y1, obj_x2, obj_y2)
    
    # Calculate patch centers and bboxes
    patch_centers = []
    patch_bboxes = []
    for py in range(grid_h):
        for px in range(grid_w):
            # Patch bbox coordinates
            patch_x1 = px / grid_w
            patch_y1 = py / grid_h
            patch_x2 = (px + 1) / grid_w
            patch_y2 = (py + 1) / grid_h
            
            # Patch center coordinates
            patch_cx = (px + 0.5) / grid_w
            patch_cy = (py + 0.5) / grid_h
            
            patch_centers.append((patch_cx, patch_cy))
            patch_bboxes.append((patch_x1, patch_y1, patch_x2, patch_y2))
    
    # Encode deltas for each patch
    bbox_deltas = torch.zeros(num_patches, 4, dtype=torch.float32)
    pos_mask = torch.zeros(num_patches, dtype=torch.bool)
    
    for i, ((patch_cx, patch_cy), patch_bbox) in enumerate(zip(patch_centers, patch_bboxes)):
        # Compute IoU between patch and object
        iou = compute_patch_object_iou(patch_bbox, obj_bbox)
        
        # Mark as positive if IoU > threshold
        if iou > iou_threshold:
            pos_mask[i] = True
            
            # Encode deltas relative to patch center
            dx = (x_c - patch_cx) / (w + 1e-6)
            dy = (y_c - patch_cy) / (h + 1e-6)
            dw = torch.log(torch.tensor(w + 1e-6))
            dh = torch.log(torch.tensor(h + 1e-6))
            
            bbox_deltas[i] = torch.tensor([dx, dy, dw, dh])

    # Fallback: if no patch was marked positive (e.g. very small bbox or rounding/augmentation
    # moved center off patch centers), mark the nearest patch to bbox center as positive.
    # This prevents having zero positives which would make the regression loss = 0 for the
    # whole sample/batch and gives the model at least one location to regress to.
    if not pos_mask.any():
        # compute distances to patch centers and pick the nearest
        centers = torch.tensor(patch_centers, dtype=torch.float32)
        bbox_center = torch.tensor([x_c, y_c], dtype=torch.float32)
        dists = torch.norm(centers - bbox_center.unsqueeze(0), dim=1)
        nearest_idx = int(torch.argmin(dists).item())
        pos_mask[nearest_idx] = True

        patch_cx, patch_cy = patch_centers[nearest_idx]
        dx = (x_c - patch_cx) / (w + 1e-6)
        dy = (y_c - patch_cy) / (h + 1e-6)
        dw = torch.log(torch.tensor(w + 1e-6))
        dh = torch.log(torch.tensor(h + 1e-6))
        bbox_deltas[nearest_idx] = torch.tensor([dx, dy, dw, dh])

    return bbox_deltas, pos_mask


def decode_patch_bbox(patch_idx: int, deltas: torch.Tensor, patch_grid_info: dict) -> torch.Tensor:
    """
    Decode bbox from patch index and deltas.
    
    Args:
        patch_idx: Index of the patch (0-7)
        deltas: (4,) - [dx, dy, dw, dh]
        patch_grid_info: Dict with grid_h, grid_w
        
    Returns:
        box: (4,) - [x_c, y_c, w, h] in normalized coordinates
    """
    grid_h = patch_grid_info['grid_h']  # 4
    grid_w = patch_grid_info['grid_w']  # 4
    
    # Convert patch index to grid coordinates
    patch_y = patch_idx // grid_w
    patch_x = patch_idx % grid_w
    
    # Patch center in normalized coordinates
    patch_cx = (patch_x + 0.5) / grid_w
    patch_cy = (patch_y + 0.5) / grid_h
    
    # Decode deltas
    dx, dy, dw_log, dh_log = deltas
    
    # Decode width/height from log-scale
    w = torch.exp(dw_log)
    h = torch.exp(dh_log)
    
    # Decode center coordinates
    x_c = patch_cx + dx * w
    y_c = patch_cy + dy * h
    
    return torch.tensor([x_c, y_c, w, h])
