#!/usr/bin/env python3
"""Create zoomed dataset from original dataset by cropping images with different area ratios."""

import argparse
import shutil
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from tqdm import tqdm


def zoom_to_bbox(
    img: np.ndarray,
    bbox_x_c: float,
    bbox_y_c: float,
    bbox_w: float,
    bbox_h: float,
    area_ratio: float
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop image with area_ratio, ensuring bbox is contained within crop.
    
    Args:
        img: Original image (H, W, C)
        bbox_x_c: Bbox center x (normalized 0-1)
        bbox_y_c: Bbox center y (normalized 0-1)
        bbox_w: Bbox width (normalized 0-1)
        bbox_h: Bbox height (normalized 0-1)
        area_ratio: Target area ratio (e.g., 0.15 for 15% of original area)
    
    Returns:
        (cropped_img, (x1, y1, x2, y2)): Cropped image and crop coordinates in original image
    """
    H, W = img.shape[:2]
    
    # Calculate target crop size based on area ratio
    target_area = H * W * area_ratio
    target_side = int(np.sqrt(target_area))
    
    # Convert normalized bbox to pixel coordinates
    bbox_x_c_px = bbox_x_c * W
    bbox_y_c_px = bbox_y_c * H
    bbox_w_px = bbox_w * W
    bbox_h_px = bbox_h * H
    
    # Calculate bbox boundaries
    bbox_x1_px = bbox_x_c_px - bbox_w_px / 2
    bbox_y1_px = bbox_y_c_px - bbox_h_px / 2
    bbox_x2_px = bbox_x_c_px + bbox_w_px / 2
    bbox_y2_px = bbox_y_c_px + bbox_h_px / 2
    
    # Ensure crop size is at least as large as bbox
    crop_w = max(target_side, int(bbox_w_px) + 20)  # Add small padding
    crop_h = max(target_side, int(bbox_h_px) + 20)
    
    # Try to center crop on bbox
    crop_x_c = bbox_x_c_px
    crop_y_c = bbox_y_c_px
    
    # Calculate crop boundaries
    crop_x1 = int(crop_x_c - crop_w / 2)
    crop_y1 = int(crop_y_c - crop_h / 2)
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h
    
    # Adjust crop boundaries to ensure bbox is fully contained
    if crop_x1 > bbox_x1_px:
        crop_x1 = int(bbox_x1_px) - 5
        crop_x2 = crop_x1 + crop_w
    if crop_y1 > bbox_y1_px:
        crop_y1 = int(bbox_y1_px) - 5
        crop_y2 = crop_y1 + crop_h
    if crop_x2 < bbox_x2_px:
        crop_x2 = int(bbox_x2_px) + 5
        crop_x1 = crop_x2 - crop_w
    if crop_y2 < bbox_y2_px:
        crop_y2 = int(bbox_y2_px) + 5
        crop_y1 = crop_y2 - crop_h
    
    # Clamp to image boundaries
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(W, crop_x2)
    crop_y2 = min(H, crop_y2)
    
    # Final crop size
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    
    # Crop image
    cropped_img = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    return cropped_img, (crop_x1, crop_y1, crop_x2, crop_y2)


def transform_bbox_to_zoomed_coords(
    bbox_x_c: float,
    bbox_y_c: float,
    bbox_w: float,
    bbox_h: float,
    img_w: int,
    img_h: int,
    crop_coords: Tuple[int, int, int, int]
) -> Tuple[float, float, float, float]:
    """
    Transform bbox coordinates from original image to zoomed image.
    
    Args:
        bbox_x_c, bbox_y_c, bbox_w, bbox_h: Normalized bbox coordinates in original image
        img_w, img_h: Original image dimensions
        crop_coords: (x1, y1, x2, y2) crop coordinates in original image
    
    Returns:
        (new_x_c, new_y_c, new_w, new_h): Normalized bbox coordinates in zoomed image
    """
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    
    # Convert normalized bbox to pixel coordinates in original image
    bbox_x_c_px = bbox_x_c * img_w
    bbox_y_c_px = bbox_y_c * img_h
    bbox_w_px = bbox_w * img_w
    bbox_h_px = bbox_h * img_h
    
    # Transform to crop coordinates
    new_x_c_px = bbox_x_c_px - crop_x1
    new_y_c_px = bbox_y_c_px - crop_y1
    
    # Normalize to new image size
    new_x_c = new_x_c_px / crop_w
    new_y_c = new_y_c_px / crop_h
    new_w = bbox_w_px / crop_w
    new_h = bbox_h_px / crop_h
    
    # Clamp to [0, 1]
    new_x_c = max(0.0, min(1.0, new_x_c))
    new_y_c = max(0.0, min(1.0, new_y_c))
    new_w = max(1e-6, min(1.0, new_w))
    new_h = max(1e-6, min(1.0, new_h))
    
    return new_x_c, new_y_c, new_w, new_h


def process_split(
    source_dir: Path,
    output_dir: Path,
    split: str,
    area_ratios: list,
    ratio_index: int
):
    """Process one split (train or val) with rotating area ratios."""
    source_img_dir = source_dir / split / "search" / "images"
    source_lbl_dir = source_dir / split / "search" / "labels"
    source_template_dir = source_dir / split / "templates"
    
    output_img_dir = output_dir / split / "search" / "images"
    output_lbl_dir = output_dir / split / "search" / "labels"
    output_template_dir = output_dir / split / "templates"
    
    # Create output directories
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_lbl_dir.mkdir(parents=True, exist_ok=True)
    output_template_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy templates (same for all)
    if source_template_dir.exists():
        for template_file in tqdm(source_template_dir.glob("*.jpg"), desc=f"Copying {split} templates"):
            shutil.copy2(template_file, output_template_dir / template_file.name)
    
    # Process images
    image_files = sorted(source_img_dir.glob("*.jpg"))
    total_images = len(image_files)
    
    print(f"\nProcessing {total_images} images for {split} split...")
    
    zoomed_count = 0
    
    for idx, img_file in enumerate(tqdm(image_files, desc=f"Processing {split} images")):
        # Copy original image
        shutil.copy2(img_file, output_img_dir / img_file.name)
        
        # Copy original label
        lbl_file = source_lbl_dir / f"{img_file.stem}.txt"
        if lbl_file.exists():
            shutil.copy2(lbl_file, output_lbl_dir / f"{img_file.stem}.txt")
        
        # Create zoomed version (rotating ratio)
        area_ratio = area_ratios[(ratio_index + idx) % len(area_ratios)]
        
        # Read image
        img = cv2.imread(str(img_file))
        if img is None:
            continue
        
        H, W = img.shape[:2]
        
        # Read label
        if not lbl_file.exists():
            continue
        
        with open(lbl_file, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            continue
        
        # Parse first bbox (should be only one)
        parts = lines[0].split()
        if len(parts) < 5:
            continue
        
        try:
            class_id = parts[0]
            bbox_x_c = float(parts[1])
            bbox_y_c = float(parts[2])
            bbox_w = float(parts[3])
            bbox_h = float(parts[4])
        except ValueError:
            continue
        
        # Zoom image
        cropped_img, crop_coords = zoom_to_bbox(
            img, bbox_x_c, bbox_y_c, bbox_w, bbox_h, area_ratio
        )
        
        # Transform bbox coordinates
        new_x_c, new_y_c, new_w, new_h = transform_bbox_to_zoomed_coords(
            bbox_x_c, bbox_y_c, bbox_w, bbox_h, W, H, crop_coords
        )
        
        # Save zoomed image
        zoomed_name = f"{img_file.stem}_zoomed_{int(area_ratio * 100)}.jpg"
        cv2.imwrite(str(output_img_dir / zoomed_name), cropped_img)
        
        # Save zoomed label
        zoomed_lbl_name = f"{img_file.stem}_zoomed_{int(area_ratio * 100)}.txt"
        with open(output_lbl_dir / zoomed_lbl_name, 'w') as f:
            f.write(f"{class_id} {new_x_c:.6f} {new_y_c:.6f} {new_w:.6f} {new_h:.6f}\n")
        
        zoomed_count += 1
    
    print(f"  Created {zoomed_count} zoomed images for {split} split")


def main():
    parser = argparse.ArgumentParser(description="Create zoomed dataset from original dataset")
    parser.add_argument(
        "--source_dir",
        type=str,
        required=True,
        help="Path to source dataset (retrieval_dataset_flat)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output dataset (retrieval_dataset_flat_zoomed)"
    )
    parser.add_argument(
        "--area_ratio1",
        type=float,
        default=0.15,
        help="First area ratio (default: 0.15)"
    )
    parser.add_argument(
        "--area_ratio2",
        type=float,
        default=0.35,
        help="Second area ratio (default: 0.35)"
    )
    parser.add_argument(
        "--area_ratio3",
        type=float,
        default=0.55,
        help="Third area ratio (default: 0.55)"
    )
    parser.add_argument(
        "--area_ratio4",
        type=float,
        default=0.75,
        help="Fourth area ratio (default: 0.75)"
    )
    args = parser.parse_args()
    
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    
    area_ratios = [args.area_ratio1, args.area_ratio2, args.area_ratio3, args.area_ratio4]
    
    print("=" * 60)
    print("Creating Zoomed Dataset")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Area ratios: {area_ratios}")
    print("=" * 60)
    
    # Process train split
    process_split(source_dir, output_dir, "train", area_ratios, ratio_index=0)
    
    # Process val split
    process_split(source_dir, output_dir, "val", area_ratios, ratio_index=0)
    
    print("\n" + "=" * 60)
    print("Dataset creation completed!")
    print("=" * 60)
    print(f"Output at: {output_dir.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

