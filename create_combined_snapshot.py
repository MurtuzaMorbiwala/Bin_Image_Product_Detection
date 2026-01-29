#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a training snapshot combining existing annotations + clustered product mappings
Merges both data sources into single-class YOLO format for training
"""

import sys
import io
import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_product_mapping(mapping_path):
    """Load the clustered product mapping JSON"""
    with open(mapping_path, 'r') as f:
        return json.load(f)


def load_existing_annotation(annotation_path):
    """
    Load existing YOLO annotation file
    Returns list of bbox tuples: (class_id, x_center, y_center, width, height)
    All coordinates should be normalized (0-1)
    """
    boxes = []
    if annotation_path.exists():
        with open(annotation_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x, y, w, h = parts
                    x, y, w, h = float(x), float(y), float(w), float(h)
                    
                    # Validate normalized coordinates (0-1)
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        print(f"  [!] Warning: Box coordinates out of range in {annotation_path.name}, line {line_num}")
                        print(f"      Values: x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}")
                        continue
                    
                    boxes.append((int(class_id), x, y, w, h))
    return boxes


def convert_detection_to_yolo(detection):
    """
    Convert a detection dict to YOLO format tuple (single-class)
    
    Args:
        detection: dict with 'bbox' [x_center, y_center, width, height]
    
    Returns:
        tuple: (0, x_center, y_center, width, height)
        None if coordinates are invalid
    """
    bbox = detection['bbox']
    x_center, y_center, width, height = bbox
    
    # Validate normalized coordinates (0-1)
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
        print(f"  [!] Warning: Clustered box coordinates out of range")
        print(f"      Values: x={x_center:.4f}, y={y_center:.4f}, w={width:.4f}, h={height:.4f}")
        return None
    
    # Always use class 0 (single-class detector)
    return (0, x_center, y_center, width, height)


def create_combined_snapshot(
    product_mapping_path="clustered_output/product_mapping.json",
    existing_annotations_dir="data/working/annotations",
    images_source_dir="data/working/images",
    output_base_dir="data/snapshots",
    snapshot_name="binsense_combined",
    dataset_name="binsensecombined"
):
    """
    Create a snapshot combining existing annotations + clustered product mappings
    
    Args:
        product_mapping_path: Path to product_mapping.json
        existing_annotations_dir: Directory with existing YOLO annotations
        images_source_dir: Directory containing source images
        output_base_dir: Base directory for snapshots
        snapshot_name: Name for this snapshot
        dataset_name: Name for Kaggle dataset
    """
    
    # Convert to Path objects
    mapping_path = Path(product_mapping_path)
    annotations_dir = Path(existing_annotations_dir)
    images_dir = Path(images_source_dir)
    snapshot_dir = Path(output_base_dir) / snapshot_name
    
    # Verify inputs exist
    if not mapping_path.exists():
        print(f"[!] Product mapping not found: {mapping_path}")
        print(f"   Will use only existing annotations")
        product_mapping = {}
    else:
        print(f"[+] Found product mapping: {mapping_path}")
        product_mapping = load_product_mapping(mapping_path)
    
    if not annotations_dir.exists():
        print(f"[!] Existing annotations directory not found: {annotations_dir}")
        print(f"   Will use only clustered mappings")
        has_existing = False
    else:
        print(f"[+] Found existing annotations: {annotations_dir}")
        has_existing = True
    
    if not images_dir.exists():
        print(f"[!] Images directory not found: {images_dir}")
        return None
    
    print(f"\n[*] Creating combined snapshot: {snapshot_name}")
    print(f"[*] Images from: {images_dir}")
    
    # Create snapshot structure
    snapshot_images = snapshot_dir / "images"
    snapshot_labels = snapshot_dir / "labels"
    
    snapshot_images.mkdir(parents=True, exist_ok=True)
    snapshot_labels.mkdir(parents=True, exist_ok=True)
    
    # Collect all image IDs from both sources
    all_image_ids = set()
    
    if product_mapping:
        all_image_ids.update(product_mapping.keys())
    
    if has_existing:
        for ann_file in annotations_dir.glob("*.txt"):
            all_image_ids.add(ann_file.stem)
    
    print(f"\n[*] Processing {len(all_image_ids)} unique images...")
    
    # Statistics
    copied_images = 0
    created_labels = 0
    total_boxes = 0
    boxes_from_existing = 0
    boxes_from_clustered = 0
    skipped_images = 0
    unique_products = set()
    product_counts = defaultdict(int)
    
    source_stats = {
        'both_sources': 0,
        'existing_only': 0,
        'clustered_only': 0
    }
    
    for image_id in sorted(all_image_ids):
        # Find corresponding image file
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG']:
            img_path = images_dir / f"{image_id}{ext}"
            if img_path.exists():
                image_file = img_path
                break
        
        if not image_file:
            print(f"  [!] Warning: Image not found for {image_id}")
            skipped_images += 1
            continue
        
        # Load existing annotations
        existing_boxes = []
        if has_existing:
            ann_path = annotations_dir / f"{image_id}.txt"
            existing_boxes = load_existing_annotation(ann_path)
        
        # Load clustered detections
        new_boxes = []
        if image_id in product_mapping:
            for detection in product_mapping[image_id]:
                new_box = convert_detection_to_yolo(detection)
                if new_box is not None:  # Only add valid boxes
                    new_boxes.append(new_box)
                    
                    # Track product statistics
                    product_id = detection['original_product_id']
                    unique_products.add(product_id)
                    product_counts[product_id] += 1
        
        # Track source statistics
        if existing_boxes and new_boxes:
            source_stats['both_sources'] += 1
        elif existing_boxes:
            source_stats['existing_only'] += 1
        elif new_boxes:
            source_stats['clustered_only'] += 1
        
        # Combine all boxes (no duplicate checking)
        all_boxes = existing_boxes + new_boxes
        
        if not all_boxes:
            continue  # Skip images with no annotations
        
        # Copy image
        shutil.copy2(image_file, snapshot_images / image_file.name)
        copied_images += 1
        
        # Write combined annotations
        label_path = snapshot_labels / f"{image_id}.txt"
        with open(label_path, 'w') as f:
            for box in all_boxes:
                class_id, x, y, w, h = box
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
        
        created_labels += 1
        total_boxes += len(all_boxes)
        boxes_from_existing += len(existing_boxes)
        boxes_from_clustered += len(new_boxes)
    
    # Create data.yaml for YOLO
    data_yaml = {
        'path': f'/kaggle/input/{dataset_name}',
        'train': 'images',
        'val': 'images',  # For now, use same for validation
        'nc': 1,  # Single class: "product"
        'names': ['product']
    }
    
    yaml_path = snapshot_dir / "data.yaml"
    with open(yaml_path, 'w') as f:
        for key, value in data_yaml.items():
            if isinstance(value, list):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, str):
                f.write(f"{key}: '{value}'\n")
            else:
                f.write(f"{key}: {value}\n")
    
    # Create metadata file
    metadata = {
        'snapshot_name': snapshot_name,
        'created': datetime.now().isoformat(),
        'images': copied_images,
        'annotations': created_labels,
        'total_boxes': total_boxes,
        'unique_products': len(unique_products),
        'classes': 1,
        'class_names': ['product'],
        'source': 'combined_existing_and_clustered',
        'skipped_images': skipped_images,
        'merge_stats': {
            'boxes_from_existing': boxes_from_existing,
            'boxes_from_clustered': boxes_from_clustered,
            'images_from_both_sources': source_stats['both_sources'],
            'images_from_existing_only': source_stats['existing_only'],
            'images_from_clustered_only': source_stats['clustered_only']
        },
        'product_diversity': {
            'unique_product_ids': sorted(list(unique_products)),
            'product_counts': dict(sorted(product_counts.items(), key=lambda x: x[1], reverse=True))
        }
    }
    
    metadata_path = snapshot_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create detailed statistics report
    stats_path = snapshot_dir / "merge_stats.txt"
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("Combined Dataset Statistics\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("DATA SOURCES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Images from both sources:      {source_stats['both_sources']:5d}\n")
        f.write(f"Images from existing only:     {source_stats['existing_only']:5d}\n")
        f.write(f"Images from clustered only:    {source_stats['clustered_only']:5d}\n")
        f.write(f"Total images:                  {copied_images:5d}\n\n")
        
        f.write("BOUNDING BOXES\n")
        f.write("-" * 60 + "\n")
        f.write(f"Boxes from existing annotations: {boxes_from_existing:5d}\n")
        f.write(f"Boxes from clustered mapping:    {boxes_from_clustered:5d}\n")
        f.write(f"Final total boxes:               {total_boxes:5d}\n")
        f.write(f"Average boxes per image:         {total_boxes/copied_images:.2f}\n\n")
        
        if unique_products:
            f.write("PRODUCT DIVERSITY\n")
            f.write("-" * 60 + "\n")
            f.write(f"Unique products: {len(unique_products)}\n")
            f.write(f"Average boxes per product: {total_boxes/len(unique_products):.2f}\n\n")
            f.write("Top 20 products by box count:\n")
            for i, (product_id, count) in enumerate(sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:20], 1):
                f.write(f"  {i:2d}. {product_id}: {count:4d} boxes\n")
    
    # Create README
    readme_content = f"""# BinSense Combined Training Snapshot

## Dataset Information
- **Snapshot Name**: {snapshot_name}
- **Created**: {metadata['created']}
- **Images**: {copied_images}
- **Total Bounding Boxes**: {total_boxes}
- **Unique Products**: {len(unique_products) if unique_products else 'N/A'}
- **Classes**: 1 (generic "product" - single-class detector)
- **Source**: Combined existing annotations + clustered product mapping

## Data Sources
This snapshot combines two data sources:

1. **Existing Annotations**: {boxes_from_existing} boxes from manual/previous annotations
2. **Clustered Mapping**: {boxes_from_clustered} boxes from new product clustering

### Merge Statistics
- Images from both sources: {source_stats['both_sources']}
- Images from existing only: {source_stats['existing_only']}
- Images from clustered only: {source_stats['clustered_only']}
- Average boxes per image: {total_boxes/copied_images:.2f}

## Structure
```
{snapshot_name}/
├── images/          # Training images
├── labels/          # YOLO format annotations (.txt)
├── data.yaml        # YOLO configuration
├── metadata.json    # Dataset metadata
├── merge_stats.txt  # Detailed merge statistics
└── README.md        # This file
```

## YOLO Format
Each .txt file contains one line per bounding box:
```
class_id x_center y_center width height
```

All values are normalized (0-1). All class_id values are 0 (single-class detector).

## Training
Use with Ultralytics YOLO:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data='data.yaml', epochs=50, imgsz=640)
```

## Notes
- This is a **single-class detector** (all products = class 0)
- Combines the best of both annotation sources
- No duplicate checking - all boxes from both sources are included
- Skipped images (if any): {skipped_images}
"""
    
    readme_path = snapshot_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n[+] Combined snapshot created successfully!")
    print(f"[*] Location: {snapshot_dir}")
    print(f"\n[*] Statistics:")
    print(f"   - Images: {copied_images}")
    print(f"   - Annotations: {created_labels}")
    print(f"   - Total boxes: {total_boxes}")
    print(f"   - Avg boxes/image: {total_boxes/copied_images:.2f}")
    print(f"\n[*] Merge Details:")
    print(f"   - Boxes from existing: {boxes_from_existing}")
    print(f"   - Boxes from clustered: {boxes_from_clustered}")
    print(f"   - Images from both: {source_stats['both_sources']}")
    print(f"   - Images from existing only: {source_stats['existing_only']}")
    print(f"   - Images from clustered only: {source_stats['clustered_only']}")
    if unique_products:
        print(f"\n[*] Product Diversity:")
        print(f"   - Unique products: {len(unique_products)}")
        print(f"   - Avg boxes/product: {total_boxes/len(unique_products):.2f}")
    if skipped_images > 0:
        print(f"\n[!] Skipped images: {skipped_images}")
    
    # Create zip file
    print(f"\n[*] Creating zip file...")
    import zipfile
    zip_path = Path(output_base_dir) / f"{snapshot_name}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in snapshot_dir.rglob('*'):
            if file_path.is_file():
                arcname = file_path.relative_to(snapshot_dir.parent)
                zipf.write(file_path, arcname)
    
    zip_size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"[+] Zip created: {zip_path} ({zip_size_mb:.2f} MB)")
    print(f"\n[*] Ready for Kaggle upload!")
    print(f"   1. Upload {snapshot_name}.zip to Kaggle")
    print(f"   2. Name the dataset: {dataset_name}")
    print(f"   3. Kaggle path will be: /kaggle/input/{dataset_name}/{snapshot_name}/")
    
    return snapshot_dir


if __name__ == "__main__":
    # Default paths - adjust as needed
    create_combined_snapshot(
        product_mapping_path="clustered_output/product_mapping.json",
        existing_annotations_dir="data/working/annotations",
        images_source_dir="data/working/images",
        output_base_dir="data/snapshots",
        snapshot_name="binsense_combined",
        dataset_name="binsensecombined"
    )