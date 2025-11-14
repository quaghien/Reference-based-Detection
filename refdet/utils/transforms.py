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
    Build image transformation pipeline with comprehensive augmentation.
    
    Augmentation includes:
    - Geometric: Random rotate ±10°, horizontal/vertical flip, perspective shift, affine
    - Color: Brightness ±0.4, contrast ±0.3, saturation ±0.3
    
    Args:
        img_size: Target image size (square)
        augment: Whether to apply data augmentation
        
    Returns:
        transform: Transformation function
    """
    # Normalization (ImageNet stats)
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def transform(image: Union[str, Path, np.ndarray], 
                  aug_params: dict = None) -> torch.Tensor:
        """
        Transform image to tensor.
        
        Args:
            image: Input image (path or numpy array)
            aug_params: Optional augmentation parameters (for sync augmentation)
            
        Returns:
            tensor: Normalized image tensor (3, img_size, img_size)
        """
        # Load image
        img = _load_image(image)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to square
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        
        # Convert to PIL for torchvision transforms
        pil = T.functional.to_pil_image(img)
        
        # Convert to tensor first (needed for geometric transforms)
        tensor = T.functional.to_tensor(pil)
        
        # Apply augmentation if enabled
        if augment:
            if aug_params is None:
                # Generate random augmentation parameters
                aug_params = {
                    'angle': random.uniform(-10, 10),  # ±10 degrees
                    'flip_h': random.random() < 0.5,   # 50% horizontal flip
                    'flip_v': random.random() < 0.3,   # 30% vertical flip (less common)
                    'brightness': random.uniform(0.6, 1.4),  # ±0.4
                    'contrast': random.uniform(0.7, 1.3),   # ±0.3
                    'saturation': random.uniform(0.7, 1.3), # ±0.3
                    'affine_angle': random.uniform(-5, 5),
                    'affine_translate': (random.uniform(-0.1, 0.1) * img_size, 
                                        random.uniform(-0.1, 0.1) * img_size),
                    'affine_scale': random.uniform(0.9, 1.1),
                    'affine_shear': (random.uniform(-5, 5), random.uniform(-5, 5)),
                }
            
            # Apply geometric augmentation
            affine_params = (
                aug_params['affine_angle'],
                aug_params['affine_translate'],
                aug_params['affine_scale'],
                aug_params['affine_shear']
            )
            tensor = apply_geometric_augmentation(
                tensor,
                angle=aug_params['angle'],
                flip_h=aug_params['flip_h'],
                flip_v=aug_params['flip_v'],
                affine_params=affine_params
            )
            
            # Convert back to PIL for color augmentation
            pil = T.functional.to_pil_image(tensor)
            
            # Apply color augmentation
            pil = TF.adjust_brightness(pil, aug_params['brightness'])
            pil = TF.adjust_contrast(pil, aug_params['contrast'])
            pil = TF.adjust_saturation(pil, aug_params['saturation'])
            
            # Convert to tensor
            tensor = T.functional.to_tensor(pil)
        else:
            # Just convert to tensor if no augmentation
            tensor = T.functional.to_tensor(pil)
        
        # Normalize
        return normalize(tensor)
    
    return transform
