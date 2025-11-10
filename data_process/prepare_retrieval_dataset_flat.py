import os
import json
import shutil
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm


VAL_PER_VIDEO = 10  # số ảnh ngẫu nhiên mỗi video_id cho split val (không loại khỏi train)


def convert_to_yolo(box, img_w, img_h):
    x1, y1, x2, y2 = map(float, box)
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + w / 2.0
    y_c = y1 + h / 2.0
    return (
        max(0.0, min(1.0, x_c / img_w)),
        max(0.0, min(1.0, y_c / img_h)),
        max(1e-6, min(1.0, w / img_w)),
        max(1e-6, min(1.0, h / img_h)),
    )


def load_annotations(annotations_path: Path) -> List[Dict]:
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("annotations.json phải là list các video record")
    return data


def ensure_dirs(root: Path):
    # Flattened structure
    (root / 'train' / 'templates').mkdir(parents=True, exist_ok=True)
    (root / 'train' / 'search' / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'train' / 'search' / 'labels').mkdir(parents=True, exist_ok=True)

    (root / 'val' / 'templates').mkdir(parents=True, exist_ok=True)
    (root / 'val' / 'search' / 'images').mkdir(parents=True, exist_ok=True)
    (root / 'val' / 'search' / 'labels').mkdir(parents=True, exist_ok=True)


def copy_templates_flat(samples_root: Path, video_id: str, dst_root: Path):
    src = samples_root / video_id / 'object_images'
    if not src.exists():
        return 0
    count = 0
    for i, img_path in enumerate(sorted(src.glob('*.jpg'))):
        name = f'{video_id}_ref_{i+1:03d}.jpg'
        for split in ['train', 'val']:
            shutil.copy2(img_path, dst_root / split / 'templates' / name)
        count += 1
    return count


def process_video_to_train(samples_root: Path, video_id: str, record: Dict, dst_root: Path) -> List[str]:
    """Extract all annotated frames to TRAIN (images+labels). Return list of base names saved."""
    video_path = samples_root / video_id / 'drone_video.mp4'
    saved_basenames: List[str] = []
    if not video_path.exists():
        print(f"Warning: missing video {video_path}")
        return saved_basenames

    frames_to_boxes: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for interval in record.get('annotations', []):
        for bb in interval.get('bboxes', []):
            try:
                fr = int(bb['frame'])
                box = (int(bb['x1']), int(bb['y1']), int(bb['x2']), int(bb['y2']))
                frames_to_boxes.setdefault(fr, []).append(box)
            except Exception:
                continue

    if not frames_to_boxes:
        return saved_basenames

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return saved_basenames
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if img_w == 0 or img_h == 0:
        print(f"Error: invalid video size for {video_path}")
        cap.release()
        return saved_basenames

    img_dir = dst_root / 'train' / 'search' / 'images'
    lbl_dir = dst_root / 'train' / 'search' / 'labels'

    for frame_idx, boxes in frames_to_boxes.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        base = f'{video_id}_frame_{frame_idx:06d}'
        img_path = img_dir / f'{base}.jpg'
        lbl_path = lbl_dir / f'{base}.txt'
        cv2.imwrite(str(img_path), frame)
        with open(lbl_path, 'w') as f:
            for (x1, y1, x2, y2) in boxes:
                x_c, y_c, w, h = convert_to_yolo((x1, y1, x2, y2), img_w, img_h)
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        saved_basenames.append(base)

    cap.release()
    return saved_basenames


def copy_random_val(dst_root: Path, basenames: List[str], per_video: int = VAL_PER_VIDEO):
    if not basenames:
        return 0
    chosen = random.sample(basenames, k=min(per_video, len(basenames)))
    src_img_dir = dst_root / 'train' / 'search' / 'images'
    src_lbl_dir = dst_root / 'train' / 'search' / 'labels'
    dst_img_dir = dst_root / 'val' / 'search' / 'images'
    dst_lbl_dir = dst_root / 'val' / 'search' / 'labels'
    count = 0
    for base in chosen:
        shutil.copy2(src_img_dir / f'{base}.jpg', dst_img_dir / f'{base}.jpg')
        shutil.copy2(src_lbl_dir / f'{base}.txt', dst_lbl_dir / f'{base}.txt')
        count += 1
    return count


def main(source_dir: str = 'train', output_dir: str = 'retrieval_dataset_flat', seed: int = 42):
    random.seed(seed)
    src_root = Path(source_dir)
    ann_path = src_root / 'annotations' / 'annotations.json'
    samples_root = src_root / 'samples'
    dst_root = Path(output_dir)

    records = load_annotations(ann_path)
    ensure_dirs(dst_root)

    rec_map = {r['video_id']: r for r in records}

    total_train, total_val, total_templates = 0, 0, 0

    print('Copying templates (flattened to train/ & val/)...')
    for vid in tqdm(rec_map.keys(), desc='Templates'):
        total_templates += copy_templates_flat(samples_root, vid, dst_root)

    print('Extracting ALL annotated frames to train (flattened images/labels)...')
    saved_map: Dict[str, List[str]] = {}
    for vid in tqdm(rec_map.keys(), desc='Videos'):
        basenames = process_video_to_train(samples_root, vid, rec_map[vid], dst_root)
        saved_map[vid] = basenames
        total_train += len(basenames)

    print('Sampling random frames per video for VAL (copied, not removed from train)...')
    for vid, basenames in tqdm(saved_map.items(), desc='Val sampling'):
        total_val += copy_random_val(dst_root, basenames, per_video=VAL_PER_VIDEO)

    print('\n--- Done ---')
    print(f"Templates copied: {total_templates}")
    print(f"Train frames:     {total_train}")
    print(f"Val frames:       {total_val}")
    print(f"Output at:        {dst_root.resolve()}")


if __name__ == '__main__':
    main()
