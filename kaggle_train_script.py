#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle Training Script for BinSense
This script runs directly on Kaggle as a Python Script (not notebook)
"""

import os
import sys
import io
from pathlib import Path
import shutil
import json

# Fix Windows console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    print("="*60)
    print("BINSENSE YOLO TRAINING ON KAGGLE")
    print("="*60)
    
    # Install dependencies
    print("\n[*] Installing dependencies...")
    # Install ultralytics first, then force downgrade numpy for matplotlib compatibility
    os.system("pip install ultralytics -q")
    os.system("pip install --force-reinstall 'numpy<2.0,>=1.26.0' -q")
    print("[+] NumPy downgraded for matplotlib compatibility")
    
    # Import after installation
    import ultralytics
    from ultralytics import YOLO
    import torch
    import numpy as np 
    
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Device Name:    {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Only'}")
    print(f"PyTorch:        {torch.__version__}")
    print(f"NumPy:          {np.__version__}")
    print(f"Ultralytics:    {ultralytics.__version__}")
    print(f"[+] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[+] GPU: {torch.cuda.get_device_name(0)}")
    
    # Setup paths - Kaggle automatically mounts datasets to /kaggle/input/
    print("\n[*] Setting up paths...")
    
    # Debug: List what's actually mounted
    print("\n[DEBUG] Contents of /kaggle/input/:")
    input_dir = Path('/kaggle/input')
    if input_dir.exists():
        for item in input_dir.iterdir():
            print(f"   - {item.name}/")
            if item.is_dir():
                for subitem in item.iterdir():
                    print(f"      - {subitem.name}")
    
    # Path to dataset - try combined first, fall back to old fixed
    DATA_PATH = None
    
    # First try: binsensecombined (new combined dataset)
    if Path('/kaggle/input/binsensecombined').exists():
        DATA_PATH = Path('/kaggle/input/binsensecombined/binsense_combined')
        if not DATA_PATH.exists():
            # Try flat structure
            DATA_PATH = Path('/kaggle/input/binsensecombined')
        print(f"[+] Using combined dataset: {DATA_PATH}")
    
    # Fallback: binsensefixed (old dataset)
    elif Path('/kaggle/input/binsensefixed').exists():
        DATA_PATH = Path('/kaggle/input/binsensefixed/binsense')
        if not DATA_PATH.exists():
            DATA_PATH = Path('/kaggle/input/binsensefixed')
        print(f"[+] Using fixed dataset: {DATA_PATH}")
    
    if DATA_PATH is None or not DATA_PATH.exists():
        print("\n[!] No BinSense dataset found!")
        print("   Expected one of:")
        print("   - /kaggle/input/binsensecombined/")
        print("   - /kaggle/input/binsensefixed/")
        print("   Make sure you've added the dataset to this kernel!")
        sys.exit(1)
    
    print(f"[+] Found dataset at: {DATA_PATH}")
    
    # Verify structure
    data_yaml = DATA_PATH / 'data.yaml'
    images_dir = DATA_PATH / 'images'
    labels_dir = DATA_PATH / 'labels'
    
    print(f"\n[*] Dataset verification:")
    print(f"   data.yaml exists: {data_yaml.exists()}")
    print(f"   images/ exists: {images_dir.exists()}")
    print(f"   labels/ exists: {labels_dir.exists()}")
    
    if not data_yaml.exists():
        print("[!] data.yaml not found!")
        sys.exit(1)
    
    if images_dir.exists():
        image_count = len(list(images_dir.glob('*.*')))
        print(f"   Images found: {image_count}")
    
    if labels_dir.exists():
        label_count = len(list(labels_dir.glob('*.txt')))
        print(f"   Labels found: {label_count}")
        
        # Count total boxes
        total_boxes = 0
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                total_boxes += len(f.readlines())
        print(f"   Total boxes: {total_boxes}")
    
    # Training configuration
    print("\n[*] Training configuration:")
    EPOCHS = int(os.environ.get('EPOCHS', 150))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '16'))
    IMG_SIZE = int(os.environ.get('IMG_SIZE', '640'))
    MODEL_SIZE = os.environ.get('MODEL_SIZE', 'n')  # n, s, m, l, x
    
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Image size: {IMG_SIZE}")
    print(f"   Model: YOLOv8{MODEL_SIZE}")
    
    # Initialize model
    print(f"\n[*] Loading YOLOv8{MODEL_SIZE} model...")
    model = YOLO(f'yolov8{MODEL_SIZE}.pt')
    
    # Train
    print("\n[*] Starting training...")
    print("-"*60)
    
    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=50,
        save=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='/kaggle/working/runs/detect',
        name='binsense_detector',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.0001,  # Lower LR for small dataset stability
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        label_smoothing=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
        verbose=True
    )
    
    print("\n[+] Training completed!")
    
    # Validate
    print("\n[*] Running validation...")
    metrics = model.val()
    
    print("\n" + "="*60)
    print("VALIDATION METRICS")
    print("="*60)
    print(f"mAP50:        {metrics.box.map50:.4f}")
    print(f"mAP50-95:     {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")
    print("="*60)
    
    # Test inference on sample image
    print("\n[*] Testing inference on sample image...")
    test_images = list(images_dir.glob('*.*'))
    if test_images:
        test_image = test_images[0]
        print(f"   Testing on: {test_image.name}")
        
        results = model.predict(test_image, conf=0.25, save=True)
        print(f"   Detections: {len(results[0].boxes)}")
        
        # Save prediction
        pred_path = '/kaggle/working/sample_prediction.jpg'
        results[0].save(pred_path)
        print(f"   Saved to: {pred_path}")
    
    # Save best weights to output
    print("\n[*] Saving model weights...")
    best_weights = '/kaggle/working/runs/detect/binsense_detector/weights/best.pt'
    last_weights = '/kaggle/working/runs/detect/binsense_detector/weights/last.pt'
    
    if Path(best_weights).exists():
        shutil.copy(best_weights, '/kaggle/working/best.pt')
        print(f"[+] Best weights: /kaggle/working/best.pt")
        print(f"   Size: {Path('/kaggle/working/best.pt').stat().st_size / 1024 / 1024:.2f} MB")
    
    if Path(last_weights).exists():
        shutil.copy(last_weights, '/kaggle/working/last.pt')
        print(f"[+] Last weights: /kaggle/working/last.pt")
    
    # Save training summary
    summary = {
        'dataset': DATA_PATH.name,
        'images': image_count if images_dir.exists() else 0,
        'labels': label_count if labels_dir.exists() else 0,
        'total_boxes': total_boxes if labels_dir.exists() else 0,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'img_size': IMG_SIZE,
        'model': f'yolov8{MODEL_SIZE}',
        'metrics': {
            'map50': float(metrics.box.map50),
            'map50_95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr)
        }
    }
    
    summary_path = '/kaggle/working/training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Training summary: {summary_path}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print("\n[*] Download these files from /kaggle/working/:")
    print("   - best.pt (trained model)")
    print("   - training_summary.json (metrics)")
    print("   - sample_prediction.jpg (test result)")
    print("\n[+] Done!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[!] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)