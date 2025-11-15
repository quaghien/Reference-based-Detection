"""Image transformation utilities for RefDet V2."""

import random
from typing import Callable, Union, Tuple, Optional
from pathlib import Path
import math

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


def _load_image(image: Union[str, Path, np.ndarray]) -> np.ndarray:
    """Load image from path or return if already numpy array."""
    if isinstance(image, np.ndarray):
        return image
    image = cv2.imread(str(image))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    return image


def transform_bbox(
    x_c: float, y_c: float, w: float, h: float,
    img_size: int,
    angle: Optional[float] = None,
    flip_h: bool = False,
    flip_v: bool = False,
    affine_params: Optional[Tuple[float, Tuple[float, float], float, Tuple[float, float]]] = None
) -> Tuple[float, float, float, float]:
    """
    Transform bounding box coordinates according to geometric augmentations.
    
    Args:
        x_c, y_c, w, h: Normalized bbox coordinates (0-1) in format (center_x, center_y, width, height)
        img_size: Image size (assumed square)
        angle: Rotation angle in degrees (applied first)
        flip_h: Whether to flip horizontally (applied after rotation)
        flip_v: Whether to flip vertically (applied after flip_h)
        affine_params: (angle, translate, scale, shear) - applied last
        
    Returns:
        (new_x_c, new_y_c, new_w, new_h): Transformed normalized bbox coordinates
    """
    # Convert to pixel coordinates
    x_c_px = x_c * img_size
    y_c_px = y_c * img_size
    w_px = w * img_size
    h_px = h * img_size
    
    # Convert center-based to corner-based for easier transformation
    x1 = x_c_px - w_px / 2
    y1 = y_c_px - h_px / 2
    x2 = x_c_px + w_px / 2
    y2 = y_c_px + h_px / 2
    
    # Get 4 corners
    corners = np.array([
        [x1, y1],  # top-left
        [x2, y1],  # top-right
        [x2, y2],  # bottom-right
        [x1, y2],  # bottom-left
    ], dtype=np.float32)
    
    center = np.array([img_size / 2, img_size / 2], dtype=np.float32)
    
    # Apply transformations in the same order as image transformation
    
    # 1. Rotation (around image center)
    if angle is not None and abs(angle) > 1e-6:
        angle_rad = math.radians(angle)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        corners = (corners - center) @ rot_matrix.T + center
    
    # 2. Horizontal flip
    if flip_h:
        corners[:, 0] = img_size - corners[:, 0]
    
    # 3. Vertical flip
    if flip_v:
        corners[:, 1] = img_size - corners[:, 1]
    
    # 4. Affine transformation
    if affine_params is not None:
        aff_angle, aff_translate, aff_scale, aff_shear = affine_params
        tx, ty = aff_translate
        
        # Build affine transformation matrix
        # Order: translate -> rotate -> scale -> shear
        aff_angle_rad = math.radians(aff_angle)
        cos_aff = math.cos(aff_angle_rad)
        sin_aff = math.sin(aff_angle_rad)
        
        # Translation
        corners = corners + np.array([tx, ty])
        
        # Rotation around center
        corners = (corners - center) @ np.array([
            [cos_aff, -sin_aff],
            [sin_aff, cos_aff]
        ]).T + center
        
        # Scale
        corners = (corners - center) * aff_scale + center
        
        # Shear (simplified - only x shear for now)
        if abs(aff_shear[0]) > 1e-6 or abs(aff_shear[1]) > 1e-6:
            shear_x, shear_y = aff_shear
            shear_matrix = np.array([
                [1, math.tan(math.radians(shear_x))],
                [math.tan(math.radians(shear_y)), 1]
            ])
            corners = (corners - center) @ shear_matrix.T + center
    
    # Convert back to center-based format
    x_min = corners[:, 0].min()
    x_max = corners[:, 0].max()
    y_min = corners[:, 1].min()
    y_max = corners[:, 1].max()
    
    new_w_px = x_max - x_min
    new_h_px = y_max - y_min
    new_x_c_px = (x_min + x_max) / 2
    new_y_c_px = (y_min + y_max) / 2
    
    # Normalize back to [0, 1] and clamp
    new_x_c = max(0.0, min(1.0, new_x_c_px / img_size))
    new_y_c = max(0.0, min(1.0, new_y_c_px / img_size))
    new_w = max(1e-6, min(1.0, new_w_px / img_size))
    new_h = max(1e-6, min(1.0, new_h_px / img_size))
    
    return new_x_c, new_y_c, new_w, new_h


def apply_geometric_augmentation(
    img: torch.Tensor,
    angle: float = None,
    flip_h: bool = None,
    flip_v: bool = None,
    affine_params: Tuple[float, Tuple[float, float], Tuple[float, float], Tuple[float, float]] = None
) -> torch.Tensor:
    """
    Apply geometric augmentation to tensor image.
    
    Args:
        img: Image tensor (C, H, W)
        angle: Rotation angle in degrees
        flip_h: Whether to flip horizontally
        flip_v: Whether to flip vertically
        affine_params: (angle, translate, scale, shear)
        
    Returns:
        Augmented image tensor
    """
    if angle is not None:
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    
    if flip_h:
        img = TF.hflip(img)
    
    if flip_v:
        img = TF.vflip(img)
    
    if affine_params is not None:
        angle, translate, scale, shear = affine_params
        img = TF.affine(img, angle=angle, translate=translate, scale=scale, shear=shear, 
                       interpolation=TF.InterpolationMode.BILINEAR, fill=0)
    
    return img


def build_transforms(img_size: int = 640, augment: bool = True) -> Callable:
    """
    Build image transformation pipeline with balanced augmentation.
    
    Pipeline: Load → Resize → [Geometric] → [Color] → Normalize
    
    Augmentation (when enabled):
      - Geometric: Rotation ±5°, H-flip 50%, V-flip 30%, Affine (translate/scale/shear)
      - Color: Brightness ±30%, Contrast ±20%, Saturation ±20%
    
    Args:
        img_size: Target square size (default 640)
        augment: Enable augmentation (default True)
        
    Returns:
        Callable that returns normalized tensor (3, img_size, img_size)
    """
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def transform(image: Union[str, Path, np.ndarray], aug_params: dict = None) -> torch.Tensor:
        """
        Transform image to normalized tensor.
        
        Args:
            image: Input image (path or numpy array)
            aug_params: Augmentation parameters (for syncing template/search)
            
        Returns:
            Normalized tensor (3, img_size, img_size)
        """
        # Load + convert BGR→RGB + resize
        img = _load_image(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        pil = T.functional.to_pil_image(img)
        tensor = T.functional.to_tensor(pil)
        
        if augment:
            if aug_params is None:
                # Generate random augmentation parameters
                aug_params = {
                    'angle': random.uniform(-5, 5),           # Rotation ±5°
                    'flip_h': random.random() < 0.5,          # H-flip 50%
                    'flip_v': random.random() < 0.3,          # V-flip 30%
                    'brightness': random.uniform(0.7, 1.3),   # ±30%
                    'contrast': random.uniform(0.8, 1.2),     # ±20%
                    'saturation': random.uniform(0.8, 1.2),   # ±20%
                    'affine_angle': random.uniform(-3, 3),
                    'affine_translate': (
                        random.uniform(-0.05, 0.05) * img_size,  # ±5% translate
                        random.uniform(-0.05, 0.05) * img_size,
                    ),
                    'affine_scale': random.uniform(0.95, 1.05),  # ±5% scale
                    'affine_shear': (random.uniform(-3, 3), random.uniform(-3, 3)),
                }
            
            # Apply geometric augmentation
            affine_params = (
                aug_params['affine_angle'],
                aug_params['affine_translate'],
                aug_params['affine_scale'],
                aug_params['affine_shear'],
            )
            tensor = apply_geometric_augmentation(
                tensor,
                angle=aug_params['angle'],
                flip_h=aug_params['flip_h'],
                flip_v=aug_params['flip_v'],
                affine_params=affine_params,
            )
            
            # Apply color augmentation
            pil = T.functional.to_pil_image(tensor)
            pil = TF.adjust_brightness(pil, aug_params['brightness'])
            pil = TF.adjust_contrast(pil, aug_params['contrast'])
            pil = TF.adjust_saturation(pil, aug_params['saturation'])
            tensor = T.functional.to_tensor(pil)
        
        # Normalize (ImageNet stats)
        return normalize(tensor)
    
    return transform
