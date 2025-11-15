"""Image transformation utilities for RefDet V2."""

import random
from typing import Callable, Union, Tuple
from pathlib import Path

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
