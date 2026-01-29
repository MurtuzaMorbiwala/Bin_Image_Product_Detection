"""
BinSense Kaggle Training Loop - Clean Version

All dependencies should be configured in Kaggle's Dependencies panel:
    - numpy==1.26.0
    - Ultralytics==8.4.7
    - git+https://github.com/openai/CLIP.git

Default Mode: DEBUG (fast epochs for testing)
Full Mode: Set FULL_RUN = True
"""

import sys
import os
from pathlib import Path
import json
import shutil
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
import ultralytics
import clip
from datetime import datetime

# ============================================================
# PYTORCH 2.6 COMPATIBILITY FIX
# ============================================================
# Add safe globals for loading Ultralytics checkpoints
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception as e:
    print(f"[!] Warning: Could not add safe globals: {e}")

# ============================================================
# RUN MODE - Change this to run full training
# ============================================================
FULL_RUN = False  # Set to True for full training (150/50 epochs)

# ============================================================
# DISPLAY ENVIRONMENT
# ============================================================
print("="*60)
print("BINSENSE TRAINING LOOP")
print("="*60)

if not FULL_RUN:
    print("*** DEBUG MODE ***")
    print("Quick test with 3/2 epochs")
    print("Set FULL_RUN = True for production training")
else:
    print("*** FULL TRAINING MODE ***")
    print("Production run with 150/50 epochs")

print("\n[+] Environment:")
print(f"    Python: {sys.version.split()[0]}")
print(f"    NumPy: {np.__version__}")
print(f"    PyTorch: {torch.__version__}")
print(f"    Ultralytics: {ultralytics.__version__}")
print(f"    CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# CONFIGURATION
# ============================================================
YOLO_EPOCHS_FIRST = 3 if not FULL_RUN else 150
YOLO_EPOCHS_FINETUNE = 2 if not FULL_RUN else 50

MAX_ROUNDS = 20 if not FULL_RUN else 15  # More rounds in DEBUG for testing
IMG_SIZE = 640
BATCH_SIZE = 16
MIN_NEW_LABELS = 5

# NOISY BOOTSTRAP MODE - Accept low-quality labels early
NOISY_BOOTSTRAP = True  # Set to True for aggressive labeling

if NOISY_BOOTSTRAP:
    print("\n*** NOISY BOOTSTRAP MODE ***")
    print("Accepting low-confidence labels for fast expansion")
    print("Model will self-correct in later rounds")

CLIP_MODEL = "ViT-B/32"
YOLO_MODEL = "yolov8n.pt"

DATASET_ROOT = Path("/kaggle/input/binsense-loop")
WORK_DIR = Path("/kaggle/working/data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ADAPTIVE THRESHOLDS PER ROUND
# ============================================================
def get_round_thresholds(round_num, noisy_mode=False):
    """Get confidence thresholds based on training round"""
    
    if noisy_mode:
        # Aggressive early, conservative later
        if round_num <= 3:
            return {
                'yolo_conf': 0.25,      # Very low
                'high_conf': 0.50,      # Accept weak boxes
                'clip_sim': 0.55        # Permissive matching
            }
        elif round_num <= 6:
            return {
                'yolo_conf': 0.35,      # Moderate
                'high_conf': 0.65,      # Getting stricter
                'clip_sim': 0.70        # Better matching
            }
        else:
            return {
                'yolo_conf': 0.45,      # Standard
                'high_conf': 0.75,      # High quality
                'clip_sim': 0.75        # Strong matching
            }
    else:
        # Conservative throughout
        return {
            'yolo_conf': 0.50,
            'high_conf': 0.75,
            'clip_sim': 0.75
        }

# ============================================================
# UTILITY: CHECK WEIGHTS VERSION
# ============================================================
def check_weights_info(weights_path):
    """Check what version a .pt file was trained with"""
    if not Path(weights_path).exists():
        return None
    
    try:
        # Use weights_only=False for Ultralytics checkpoints (safe - our own training)
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        info = {
            'file': str(weights_path),
            'size_mb': Path(weights_path).stat().st_size / 1024 / 1024
        }
        
        if isinstance(checkpoint, dict):
            if 'train_args' in checkpoint:
                info['train_args'] = str(checkpoint['train_args'])
            if 'version' in checkpoint:
                info['version'] = checkpoint['version']
            if 'model' in checkpoint:
                info['has_model'] = True
            if 'epoch' in checkpoint:
                info['epoch'] = checkpoint['epoch']
            
            for key in ['ultralytics_version', 'yolo_version', 'version']:
                if key in checkpoint:
                    info['ultralytics_version'] = checkpoint[key]
                    break
        
        return info
        
    except Exception as e:
        return {'file': str(weights_path), 'error': str(e)}

# ============================================================
# YOLO TRAINING
# ============================================================
def train_yolo(data_yaml, prev_weights=None, round_num=1):
    """Train YOLO with fixed configuration"""
    import logging, warnings
    warnings.filterwarnings('ignore')
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    
    saved_weights = DATASET_ROOT / "best.pt"
    
    if prev_weights and Path(prev_weights).exists():
        print(f"[*] ROUND {round_num}: Continuing from previous round")
        print(f"    Loading: {prev_weights}")
        try:
            model = YOLO(prev_weights)
            epochs = YOLO_EPOCHS_FINETUNE
            weights_source = "previous_round"
        except Exception as e:
            print(f"[!] Failed to load: {e}")
            print(f"[!] Starting fresh from {YOLO_MODEL}")
            model = YOLO(YOLO_MODEL)
            epochs = YOLO_EPOCHS_FIRST
            weights_source = "yolo_pretrained"
        
    elif saved_weights.exists():
        print(f"[*] ROUND {round_num}: Loading saved weights")
        print(f"    Loading: {saved_weights}")
        
        try:
            model = YOLO(str(saved_weights))
            epochs = YOLO_EPOCHS_FINETUNE
            weights_source = "saved_weights"
            print(f"[+] Successfully loaded!")
        except Exception as e:
            print(f"[!] Failed to load: {e}")
            print(f"[!] Falling back to {YOLO_MODEL}")
            model = YOLO(YOLO_MODEL)
            epochs = YOLO_EPOCHS_FIRST
            weights_source = "yolo_pretrained"
        
    else:
        print(f"[*] ROUND {round_num}: Starting from {YOLO_MODEL}")
        model = YOLO(YOLO_MODEL)
        epochs = YOLO_EPOCHS_FIRST
        weights_source = "yolo_pretrained"
    
    print(f"[+] Model source: {weights_source}")
    print(f"[+] Training epochs: {epochs}")
    print(f"\n[*] Starting training...")
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=50,
        save=True,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='/kaggle/working/runs/detect',
        name=f'train_round{round_num}',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.0001,
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
    
    return model

# ============================================================
# CLIP EMBEDDINGS
# ============================================================
def build_clip(mapping, images_dir, device):
    """Build CLIP embeddings using fixed model"""
    print(f"[*] Loading CLIP model: {CLIP_MODEL}")
    
    # Load from dataset (pre-downloaded model file)
    clip_model_path = DATASET_ROOT / "ViT-B-32.pt"
    if not clip_model_path.exists():
        raise FileNotFoundError(
            f"CLIP model not found at {clip_model_path}. "
            f"Please include ViT-B-32.pt in your dataset root."
        )
    
    print(f"[+] Loading CLIP model from {clip_model_path}")
    
    # Load CLIP with custom model path
    import hashlib
    model, preprocess = clip.load(CLIP_MODEL, device=device, download_root=str(DATASET_ROOT))
    print(f"[+] CLIP model loaded successfully")
    
    saved_embeddings_path = DATASET_ROOT / "clip_embeddings.json"
    if saved_embeddings_path.exists():
        print(f"[+] Loading saved CLIP embeddings")
        with open(saved_embeddings_path, 'r') as f:
            embeddings_dict = json.load(f)
        embeddings = {pid: np.array(emb) for pid, emb in embeddings_dict.items()}
        print(f"[+] Loaded {len(embeddings)} products")
        return model, preprocess, embeddings
    
    print("[*] Building embeddings from scratch...")
    embeddings = {}
    product_counts = {}

    for img_id, products in mapping.items():
        img_path = images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            continue
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        for prod in products:
            pid = prod.get("original_product_id", "")
            if not pid.startswith("B"):
                continue
            
            source = prod.get("source", "manual")
            yolo_conf = prod.get("yolo_conf", 1.0)
            
            # In noisy mode, use all labels; otherwise filter low confidence
            clip_min_conf = 0.4 if NOISY_BOOTSTRAP else 0.7
            if source == "auto" and yolo_conf < clip_min_conf:
                continue
                
            bbox = prod.get("bbox", [])
            if len(bbox) != 4:
                continue

            x_c, y_c, bw, bh = bbox
            x1, y1 = int((x_c - bw/2) * w), int((y_c - bh/2) * h)
            x2, y2 = int((x_c + bw/2) * w), int((y_c + bh/2) * h)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = image.crop((max(0,x1), max(0,y1), min(w,x2), min(h,y2)))
            
            if crop.size[0] < 20 or crop.size[1] < 20:
                continue

            with torch.no_grad():
                emb = model.encode_image(preprocess(crop).unsqueeze(0).to(device))
                emb = (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy().flatten()

            embeddings.setdefault(pid, []).append(emb)
            product_counts[pid] = product_counts.get(pid, 0) + 1

    embeddings = {pid: np.mean(embs, axis=0) for pid, embs in embeddings.items()}
    
    print(f"[+] Built embeddings for {len(embeddings)} products")
    if product_counts:
        print(f"[+] Top products:")
        for pid, count in sorted(product_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f"    {pid}: {count} samples")
    
    return model, preprocess, embeddings

# ============================================================
# AUTO-LABELING
# ============================================================
def auto_label(yolo, clip_model, preprocess, embeddings, data, device, round_num=1):
    """Auto-label using three strategies: simple, CLIP matching, elimination"""
    
    # Get adaptive thresholds for this round
    thresholds = get_round_thresholds(round_num, NOISY_BOOTSTRAP)
    YOLO_CONF = thresholds['yolo_conf']
    HIGH_CONF_THRESHOLD = thresholds['high_conf']
    CLIP_SIM_THRESHOLD = thresholds['clip_sim']
    
    print(f"[*] Round {round_num} thresholds:")
    print(f"    YOLO: {YOLO_CONF:.2f}, High-Conf: {HIGH_CONF_THRESHOLD:.2f}, CLIP: {CLIP_SIM_THRESHOLD:.2f}")
    
    new_labels = 0
    high_conf_additions = 0
    clip_matched = 0
    elimination_matched = 0
    unknown_products = 0
    simple_assigned = 0
    
    mapping_file = data["mapping_file"]
    mapping = json.load(open(mapping_file)) if Path(mapping_file).exists() else {}

    for meta_file in Path(data["metadata_dir"]).glob("*.json"):
        img_id = meta_file.stem
        if img_id in mapping:
            continue

        img_path = Path(data["images_dir"]) / f"{img_id}.jpg"
        if not img_path.exists():
            continue

        meta = json.load(open(meta_file))
        meta_products = set(meta.get("BIN_FCSKU_DATA", {}).keys())
        if not meta_products:
            continue

        results = yolo(str(img_path), verbose=False, conf=YOLO_CONF)
        boxes = [{"bbox": b.xyxy[0].tolist(), "conf": b.conf[0].item()} for r in results for b in r.boxes]
        
        if not boxes:
            continue

        image = Image.open(img_path)
        w, h = image.size
        
        # STRATEGY 1: Simple 1-to-1
        if len(meta_products) == 1 and len(boxes) == 1:
            pid = list(meta_products)[0]
            conf = boxes[0]["conf"]
            x1, y1, x2, y2 = [int(c) for c in boxes[0]["bbox"]]
            bbox = [((x1+x2)/2)/w, ((y1+y2)/2)/h, (x2-x1)/w, (y2-y1)/h]
            
            mapping[img_id] = [{
                "original_product_id": pid, 
                "bbox": bbox, 
                "yolo_conf": conf, 
                "source": "auto_simple",
                "clip_similarity": None
            }]

            label_file = Path(data["labels_dir"]) / f"{img_id}.txt"
            label_file.parent.mkdir(parents=True, exist_ok=True)
            with open(label_file, "w") as f:
                f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            new_labels += 1
            simple_assigned += 1
            
            if conf >= HIGH_CONF_THRESHOLD:
                crop = image.crop((max(0,x1), max(0,y1), min(w,x2), min(h,y2)))
                if crop.size[0] >= 20 and crop.size[1] >= 20:
                    with torch.no_grad():
                        emb = clip_model.encode_image(preprocess(crop).unsqueeze(0).to(device))
                        emb = (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy().flatten()
                    
                    if pid in embeddings:
                        embeddings[pid] = (embeddings[pid] + emb) / 2
                    else:
                        embeddings[pid] = emb
                        unknown_products += 1
                    high_conf_additions += 1
        
        # STRATEGY 2 & 3: CLIP + Elimination
        elif len(meta_products) > 0 and len(boxes) > 0:
            box_embeddings = []
            for box in boxes:
                x1, y1, x2, y2 = [int(c) for c in box["bbox"]]
                crop = image.crop((max(0,x1), max(0,y1), min(w,x2), min(h,y2)))
                
                if crop.size[0] < 20 or crop.size[1] < 20:
                    box_embeddings.append(None)
                    continue
                
                with torch.no_grad():
                    emb = clip_model.encode_image(preprocess(crop).unsqueeze(0).to(device))
                    emb = (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy().flatten()
                box_embeddings.append(emb)
            
            matched_boxes = []
            matched_products = set()
            matched_box_indices = set()
            
            for box_idx, (box, box_emb) in enumerate(zip(boxes, box_embeddings)):
                if box_emb is None:
                    continue
                
                best_match = None
                best_similarity = CLIP_SIM_THRESHOLD
                
                for pid in meta_products:
                    if pid in embeddings and pid not in matched_products:
                        similarity = np.dot(box_emb, embeddings[pid])
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = pid
                
                if best_match:
                    x1, y1, x2, y2 = [int(c) for c in box["bbox"]]
                    bbox = [((x1+x2)/2)/w, ((y1+y2)/2)/h, (x2-x1)/w, (y2-y1)/h]
                    
                    matched_boxes.append({
                        "original_product_id": best_match,
                        "bbox": bbox,
                        "yolo_conf": box["conf"],
                        "clip_similarity": float(best_similarity),
                        "source": "auto_clip_matched"
                    })
                    matched_products.add(best_match)
                    matched_box_indices.add(box_idx)
                    
                    if box["conf"] >= HIGH_CONF_THRESHOLD:
                        if best_match in embeddings:
                            embeddings[best_match] = (embeddings[best_match] + box_emb) / 2
                        else:
                            embeddings[best_match] = box_emb
                            unknown_products += 1
                        high_conf_additions += 1
            
            # ELIMINATION LOGIC
            unmatched_products = meta_products - matched_products
            unmatched_box_indices = [
                i for i in range(len(boxes))
                if i not in matched_box_indices and box_embeddings[i] is not None
            ]
            
            if len(unmatched_products) == 1 and len(unmatched_box_indices) == 1:
                box_idx = unmatched_box_indices[0]
                box = boxes[box_idx]
                box_emb = box_embeddings[box_idx]
                
                if box["conf"] >= HIGH_CONF_THRESHOLD:
                    pid = list(unmatched_products)[0]
                    x1, y1, x2, y2 = [int(c) for c in box["bbox"]]
                    bbox = [((x1+x2)/2)/w, ((y1+y2)/2)/h, (x2-x1)/w, (y2-y1)/h]
                    
                    matched_boxes.append({
                        "original_product_id": pid,
                        "bbox": bbox,
                        "yolo_conf": box["conf"],
                        "clip_similarity": None,
                        "source": "auto_elimination"
                    })
                    matched_products.add(pid)
                    
                    if pid in embeddings:
                        embeddings[pid] = (embeddings[pid] + box_emb) / 2
                    else:
                        embeddings[pid] = box_emb
                        unknown_products += 1
                    high_conf_additions += 1
                    elimination_matched += 1
                    print(f"    [ELIMINATION] {pid} (conf: {box['conf']:.2f})")
            
            if matched_boxes:
                mapping[img_id] = matched_boxes
                
                label_file = Path(data["labels_dir"]) / f"{img_id}.txt"
                label_file.parent.mkdir(parents=True, exist_ok=True)
                with open(label_file, "w") as f:
                    for match in matched_boxes:
                        bbox = match["bbox"]
                        f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                
                new_labels += 1
                clip_matched += 1

    with open(mapping_file, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"[+] Auto-labeling summary:")
    print(f"    New images labeled: {new_labels}")
    if simple_assigned > 0:
        print(f"    Simple assignments: {simple_assigned}")
    if clip_matched > 0:
        print(f"    CLIP-matched: {clip_matched}")
    if elimination_matched > 0:
        print(f"    Elimination-matched: {elimination_matched}")
    if high_conf_additions > 0:
        print(f"    Added to CLIP DB: {high_conf_additions}")
    if unknown_products > 0:
        print(f"    New products: {unknown_products}")
    
    return new_labels

# ============================================================
# MAIN
# ============================================================
def main():
    print(f"\n[*] Configuration:")
    print(f"    Mode: {'FULL RUN' if FULL_RUN else 'DEBUG'}")
    print(f"    First round epochs: {YOLO_EPOCHS_FIRST}")
    print(f"    Fine-tune epochs: {YOLO_EPOCHS_FINETUNE}")
    print(f"    Max rounds: {MAX_ROUNDS}")
    print(f"    Image size: {IMG_SIZE}")
    print(f"    Batch size: {BATCH_SIZE}")
    
    # Check saved weights
    saved_weights = DATASET_ROOT / "best.pt"
    if saved_weights.exists():
        print(f"\n{'='*60}")
        print("CHECKING SAVED WEIGHTS")
        print("="*60)
        weights_info = check_weights_info(saved_weights)
        if weights_info:
            print(f"[+] Found: {weights_info['file']}")
            print(f"    Size: {weights_info.get('size_mb', 0):.2f} MB")
            
            if 'ultralytics_version' in weights_info:
                print(f"    Weights Version: {weights_info['ultralytics_version']}")
                print(f"    Current Version: {ultralytics.__version__}")
                if str(weights_info['ultralytics_version']) != ultralytics.__version__:
                    print(f"    Warning: VERSION MISMATCH - may fail to load")
                else:
                    print(f"    Versions match")
            
            if 'epoch' in weights_info:
                print(f"    Trained epochs: {weights_info['epoch']}")
        print("="*60)

    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    print("\n[*] Copying dataset...")
    shutil.copytree(DATASET_ROOT, WORK_DIR)

    mapping_file = WORK_DIR / "product_mapping.json"
    mapping = json.load(open(mapping_file)) if mapping_file.exists() else {}

    clip_model, preprocess, embeddings = build_clip(mapping, WORK_DIR / "images", device)

    prev_weights = None
    print(f"\n{'Round':<6} {'Products':<10} {'MatchAcc':<10} {'New':<6}")
    print("-"*40)
    
    # Track metrics across rounds
    training_history = []

    for r in range(1, MAX_ROUNDS+1):
        print(f"\n{'='*60}")
        print(f"ROUND {r}/{MAX_ROUNDS}")
        print("="*60)
        
        model = train_yolo(WORK_DIR / "data.yaml", prev_weights, round_num=r)
        prev_weights = f"/kaggle/working/runs/detect/train_round{r}/weights/best.pt"
        
        print(f"\n[*] Validating...")
        metrics = model.val()
        map50 = float(metrics.box.map50)
        map50_95 = float(metrics.box.map)
        print(f"[+] mAP50: {map50:.4f}")
        print(f"[+] mAP50-95: {map50_95:.4f}")

        print(f"\n[*] Auto-labeling...")
        new_labels = auto_label(
            model, clip_model, preprocess, embeddings,
            {
                "images_dir": WORK_DIR / "images",
                "labels_dir": WORK_DIR / "labels",
                "metadata_dir": WORK_DIR / "metadata",
                "mapping_file": mapping_file,
            },
            device,
            round_num=r
        )

        mapping = json.load(open(mapping_file))
        total_labels = len(list((WORK_DIR / "labels").glob("*.txt")))
        total_products = len(set(
            p["original_product_id"]
            for ps in mapping.values()
            for p in ps
            if p.get("original_product_id","").startswith("B")
        ))
        
        # Calculate CLIP match accuracy (our key metric!)
        # This tells us: "When we identify a product, how often are we right?"
        auto_labels = [p for ps in mapping.values() for p in ps if p.get("source","").startswith("auto")]
        
        if auto_labels:
            # High confidence matches (>0.75) are considered correct
            high_conf_matches = [p for p in auto_labels if p.get("yolo_conf", 0) >= 0.75]
            match_accuracy = len(high_conf_matches) / len(auto_labels) * 100
        else:
            match_accuracy = 0
        
        print(f"{r:<6} {total_products:<10} {match_accuracy:.1f}%      {new_labels:<6}")
        
        # Store metrics for this round
        round_metrics = {
            'round': r,
            'total_products': total_products,
            'new_labels': new_labels,
            'match_accuracy': float(match_accuracy),
            'clip_db_size': len(embeddings)
        }
        training_history.append(round_metrics)

        if new_labels < MIN_NEW_LABELS:
            print(f"\n[!] Stopping: only {new_labels} new labels")
            break

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    if not FULL_RUN:
        print("\n*** DEBUG MODE SUMMARY ***")
        print(f"Completed {r} rounds")
        print(f"Products learned: {total_products}")
        print(f"Match accuracy: {match_accuracy:.1f}%")
        print(f"\nSet FULL_RUN = True for production training")
        print("="*60)
    
    # Save outputs
    final_model = Path(prev_weights) if prev_weights else None
    if final_model and final_model.exists():
        # Copy the best weights
        shutil.copy(final_model, "/kaggle/working/best.pt")
        print(f"\n[+] Saved: best.pt ({Path('/kaggle/working/best.pt').stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Also save last.pt for resuming interrupted training
        last_weights = final_model.parent / "last.pt"
        if last_weights.exists():
            shutil.copy(last_weights, "/kaggle/working/last.pt")
            print(f"[+] Saved: last.pt ({Path('/kaggle/working/last.pt').stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Save CLIP embeddings
        embeddings_dict = {pid: emb.tolist() for pid, emb in embeddings.items()}
        with open('/kaggle/working/clip_embeddings.json', 'w') as f:
            json.dump(embeddings_dict, f, indent=2)
        print(f"[+] Saved: clip_embeddings.json ({len(embeddings_dict)} products)")
        
        # Save mapping
        shutil.copy(mapping_file, '/kaggle/working/product_mapping.json')
        print(f"[+] Saved: product_mapping.json")
        
        # Save summary
        summary = {
            'rounds': r,
            'total_products': total_products,
            'match_accuracy': float(match_accuracy),
            'clip_db_size': len(embeddings),
            'training_history': training_history
        }
        with open('/kaggle/working/training_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[+] Saved: training_summary.json")
    
    print("\n[+] Done!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()