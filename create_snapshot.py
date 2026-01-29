#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a training snapshot from annotated data
Prepares data for Kaggle upload in YOLO format
"""

import sys
import io
import os
import shutil
import json
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def create_snapshot():
    """Create a snapshot of annotated data for training"""
    
    # Paths
    working_dir = Path("data/working")
    dataset_name = "binsensefixed"  # Kaggle dataset name
    snapshot_name = "binsense"  # Folder inside the zip
    snapshot_dir = Path(f"data/snapshots/{snapshot_name}")
    
    images_dir = working_dir / "images"
    annotations_dir = working_dir / "annotations"
    metadata_dir = working_dir / "metadata"
    
    # Create snapshot structure
    snapshot_images = snapshot_dir / "images"
    snapshot_labels = snapshot_dir / "labels"
    
    snapshot_images.mkdir(parents=True, exist_ok=True)
    snapshot_labels.mkdir(parents=True, exist_ok=True)
    
    # Get list of annotated images
    annotation_files = list(annotations_dir.glob("*.txt"))
    
    if not annotation_files:
        print(" No annotations found!")
        return
    
    print(f" Creating snapshot with {len(annotation_files)} annotated images...")
    
    # Copy annotated images and labels
    copied_images = 0
    copied_labels = 0
    total_boxes = 0
    
    for ann_file in annotation_files:
        image_id = ann_file.stem
        
        # Find corresponding image
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            img_path = images_dir / f"{image_id}{ext}"
            if img_path.exists():
                image_file = img_path
                break
        
        if not image_file:
            print(f"  Warning: Image not found for {image_id}")
            continue
        
        # Copy image
        shutil.copy2(image_file, snapshot_images / image_file.name)
        copied_images += 1
        
        # Copy annotation and convert all class IDs to 0 (single-class detector)
        with open(ann_file, 'r') as f:
            lines = f.readlines()
        
        with open(snapshot_labels / ann_file.name, 'w') as f:
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    # Change class_id to 0, keep bbox coordinates
                    f.write(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")
                    total_boxes += 1
        
        copied_labels += 1
    
    # Create data.yaml for YOLO
    # Path will be /kaggle/input/binsensefixed on Kaggle (flat structure)
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
        'annotations': copied_labels,
        'total_boxes': total_boxes,
        'classes': 1,
        'class_names': ['product']
    }
    
    metadata_path = snapshot_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create README
    readme_content = f"""# BinSense Training Snapshot

## Dataset Information
- **Snapshot Name**: {snapshot_name}
- **Created**: {metadata['created']}
- **Images**: {copied_images}
- **Total Bounding Boxes**: {total_boxes}
- **Classes**: 1 (generic "product")

## Structure
```
{snapshot_name}/
 images/          # Training images
 labels/          # YOLO format annotations (.txt)
 data.yaml        # YOLO configuration
 metadata.json    # Dataset metadata
```

## YOLO Format
Each .txt file contains one line per bounding box:
```
class_id x_center y_center width height
```

All values are normalized (0-1).

## Training
Use with Ultralytics YOLO:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(data='data.yaml', epochs=50, imgsz=640)
```
"""
    
    readme_path = snapshot_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n Snapshot created successfully!")
    print(f" Location: {snapshot_dir}")
    print(f" Statistics:")
    print(f"   - Images: {copied_images}")
    print(f"   - Annotations: {copied_labels}")
    print(f"   - Total boxes: {total_boxes}")
    print(f"   - Avg boxes/image: {total_boxes/copied_images:.2f}")
    
    # Create zip file
    print(f"\n Creating zip file...")
    import zipfile
    zip_path = Path(f"data/snapshots/{snapshot_name}.zip")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in snapshot_dir.rglob('*'):
            if file_path.is_file():
                # Keep binsense folder structure in zip
                # So Kaggle extracts to /kaggle/input/binsensefixed/binsense/
                arcname = file_path.relative_to(snapshot_dir.parent)
                zipf.write(file_path, arcname)
    
    zip_size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f" Zip created: {zip_path} ({zip_size_mb:.2f} MB)")
    print(f"\n Ready for Kaggle upload!")
    print(f"   1. Upload {snapshot_name}.zip to Kaggle")
    print(f"   2. Name the dataset: {dataset_name}")
    print(f"   3. Kaggle path will be: /kaggle/input/{dataset_name}/{snapshot_name}/")
    
    return snapshot_dir

if __name__ == "__main__":
    create_snapshot()
