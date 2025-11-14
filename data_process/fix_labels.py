#!/usr/bin/env python3
"""Script to fix label files: ensure each file has only 1 unique bbox."""

import argparse
from pathlib import Path
from tqdm import tqdm


def fix_label_file(label_path: Path) -> tuple[bool, str]:
    """
    Fix a single label file to have only 1 unique bbox.
    
    Returns:
        (changed, message): (bool, str) - Whether file was changed and message
    """
    if not label_path.exists():
        return False, "File not found"
    
    # Read all lines
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    if not lines:
        # Empty file - write empty
        with open(label_path, 'w') as f:
            pass
        return True, "Empty file (cleaned)"
    
    # Parse bboxes
    bboxes = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            try:
                class_id = parts[0]
                x_c, y_c, w, h = map(float, parts[1:5])
                bboxes.append((class_id, x_c, y_c, w, h))
            except (ValueError, IndexError):
                continue
    
    if not bboxes:
        # No valid bboxes - write empty
        with open(label_path, 'w') as f:
            pass
        return True, "No valid bboxes (cleaned)"
    
    # Get first unique bbox (chỉ lấy bbox đầu tiên)
    unique_bbox = bboxes[0]
    
    # Format expected line
    expected_line = f"{unique_bbox[0]} {unique_bbox[1]:.6f} {unique_bbox[2]:.6f} {unique_bbox[3]:.6f} {unique_bbox[4]:.6f}\n"
    
    # Check if file needs fixing
    needs_fix = False
    if len(bboxes) > 1:
        needs_fix = True
    elif len(lines) > 1:
        needs_fix = True
    elif len(lines) == 1 and lines[0].strip() != expected_line.strip():
        needs_fix = True
    
    if not needs_fix:
        return False, "OK"
    
    # Write fixed label (only 1 bbox)
    with open(label_path, 'w') as f:
        f.write(expected_line)
    
    if len(bboxes) > 1:
        return True, f"Fixed: {len(bboxes)} bboxes → 1 bbox (kept first)"
    elif len(lines) > 1:
        return True, f"Fixed: {len(lines)} lines → 1 line (removed duplicates)"
    else:
        return True, "Fixed: formatted"


def fix_labels_directory(labels_dir: Path, dry_run: bool = False):
    """Fix all label files in a directory."""
    label_files = sorted(labels_dir.glob("*.txt"))
    
    if not label_files:
        print(f"No label files found in {labels_dir}")
        return
    
    print(f"Processing {len(label_files)} label files in {labels_dir}...")
    
    fixed_count = 0
    error_count = 0
    empty_count = 0
    
    for label_path in tqdm(label_files, desc="Fixing labels", ncols=100):
        try:
            changed, message = fix_label_file(label_path)
            if changed:
                fixed_count += 1
                if "empty" in message.lower() or "no valid" in message.lower():
                    empty_count += 1
                if not dry_run:
                    tqdm.write(f"  {label_path.name}: {message}")
                else:
                    tqdm.write(f"  [DRY RUN] {label_path.name}: {message}")
        except Exception as e:
            error_count += 1
            tqdm.write(f"  ERROR {label_path.name}: {e}")
    
    print(f"\nSummary:")
    print(f"  Total files: {len(label_files)}")
    print(f"  Fixed: {fixed_count}")
    print(f"  Empty/Invalid: {empty_count}")
    print(f"  Errors: {error_count}")
    if dry_run:
        print(f"\n  [DRY RUN] No files were modified. Run without --dry_run to apply changes.")


def main():
    parser = argparse.ArgumentParser(description="Fix label files to have only 1 bbox per file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to retrieval_dataset_flat root")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be fixed without making changes")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Fix train labels
    train_labels_dir = data_dir / "train" / "search" / "labels"
    if train_labels_dir.exists():
        print("=" * 60)
        print("Fixing TRAIN labels...")
        print("=" * 60)
        fix_labels_directory(train_labels_dir, dry_run=args.dry_run)
    else:
        print(f"Warning: {train_labels_dir} not found")
    
    # Fix val labels
    val_labels_dir = data_dir / "val" / "search" / "labels"
    if val_labels_dir.exists():
        print("\n" + "=" * 60)
        print("Fixing VAL labels...")
        print("=" * 60)
        fix_labels_directory(val_labels_dir, dry_run=args.dry_run)
    else:
        print(f"Warning: {val_labels_dir} not found")
    
    print("\n" + "=" * 60)
    print("Label fixing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

