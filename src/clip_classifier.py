"""CLIP-based product classifier for BinSense
Uses CLIP to identify specific products from YOLO-detected bounding boxes"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional

# Will be imported after checking installation
clip = None
clip_model = None
clip_preprocess = None
device = None


def ensure_clip_installed():
    """Install CLIP if not available"""
    global clip
    try:
        import clip as clip_module
        clip = clip_module
        return True
    except ImportError:
        print("Installing CLIP...")
        import subprocess
        subprocess.run(["pip", "install", "git+https://github.com/openai/CLIP.git"], check=True)
        import clip as clip_module
        clip = clip_module
        return True


def load_clip_model(model_name: str = "ViT-B/32"):
    """Load CLIP model"""
    global clip_model, clip_preprocess, device
    
    ensure_clip_installed()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model '{model_name}' on {device}...")
    
    clip_model, clip_preprocess = clip.load(model_name, device=device)
    print("CLIP model loaded!")
    
    return clip_model, clip_preprocess


def get_image_embedding(image: Image.Image) -> np.ndarray:
    """Get CLIP embedding for an image"""
    global clip_model, clip_preprocess, device
    
    if clip_model is None:
        load_clip_model()
    
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
    
    return embedding.cpu().numpy().flatten()


def crop_from_yolo_box(image: Image.Image, bbox: list) -> Image.Image:
    """Crop image using YOLO format bbox [x_center, y_center, width, height]"""
    width, height = image.size
    
    x_center, y_center, box_w, box_h = bbox
    
    # Convert normalized coords to pixels
    x1 = int((x_center - box_w/2) * width)
    y1 = int((y_center - box_h/2) * height)
    x2 = int((x_center + box_w/2) * width)
    y2 = int((y_center + box_h/2) * height)
    
    # Clamp to image bounds
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    
    return image.crop((x1, y1, x2, y2))


class ProductDatabase:
    """Database of product embeddings for classification"""
    
    def __init__(self, embeddings_path: Optional[str] = None):
        self.embeddings = {}  # product_id -> list of embeddings
        self.embeddings_path = embeddings_path or "data/product_embeddings.npz"
        
    def add_product_embedding(self, product_id: str, embedding: np.ndarray):
        """Add an embedding for a product"""
        if product_id not in self.embeddings:
            self.embeddings[product_id] = []
        self.embeddings[product_id].append(embedding)
    
    def build_from_annotations(self, 
                                images_dir: str = "data/snapshots/binsense/images",
                                mapping_file: str = "data/product_mapping.json"):
        """Build embeddings database from annotated training data"""
        
        images_path = Path(images_dir)
        
        # Load product mapping
        with open(mapping_file, 'r') as f:
            product_mapping = json.load(f)
        
        print(f"Building product database from {len(product_mapping)} images...")
        
        total_crops = 0
        for image_id, products in product_mapping.items():
            # Find image file
            image_file = images_path / f"{image_id}.jpg"
            if not image_file.exists():
                continue
            
            image = Image.open(image_file).convert('RGB')
            
            for product in products:
                product_id = product['original_product_id']
                bbox = product['bbox']
                
                # Crop and get embedding
                crop = crop_from_yolo_box(image, bbox)
                embedding = get_image_embedding(crop)
                
                self.add_product_embedding(product_id, embedding)
                total_crops += 1
        
        print(f"Created embeddings for {len(self.embeddings)} products from {total_crops} crops")
        return self
    
    def save(self, path: Optional[str] = None):
        """Save embeddings to file"""
        path = path or self.embeddings_path
        
        # Convert to arrays for saving
        save_data = {}
        for product_id, emb_list in self.embeddings.items():
            save_data[f"product_{product_id}"] = np.array(emb_list)
        
        np.savez(path, **save_data)
        print(f"Saved embeddings to {path}")
    
    def load(self, path: Optional[str] = None):
        """Load embeddings from file"""
        path = path or self.embeddings_path
        
        if not Path(path).exists():
            print(f"No embeddings file found at {path}")
            return self
        
        data = np.load(path)
        self.embeddings = {}
        
        for key in data.files:
            product_id = key.replace("product_", "")
            self.embeddings[product_id] = list(data[key])
        
        print(f"Loaded embeddings for {len(self.embeddings)} products")
        return self
    
    def classify(self, embedding: np.ndarray, top_k: int = 3) -> list:
        """Classify an embedding against the database
        
        Returns list of (product_id, confidence) tuples
        """
        if not self.embeddings:
            return []
        
        # Compute similarity to all product embeddings
        scores = {}
        
        for product_id, emb_list in self.embeddings.items():
            # Average similarity across all embeddings for this product
            similarities = []
            for ref_emb in emb_list:
                sim = np.dot(embedding, ref_emb)  # Cosine similarity (normalized)
                similarities.append(sim)
            
            scores[product_id] = max(similarities)  # Best match
        
        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_scores[:top_k]


class CLIPClassifier:
    """Main classifier combining YOLO detection with CLIP classification"""
    
    def __init__(self, yolo_model_path: str = "kaggle_output/best.pt"):
        self.yolo_model_path = yolo_model_path
        self.yolo_model = None
        self.product_db = ProductDatabase()
    
    def export_for_training(self, image_path: str, detections: list,
                            output_dir: str = "data/working/annotations",
                            mapping_file: str = "data/product_mapping.json") -> dict:
        """
        Export validated detections as YOLO training annotations.
        
        Creates:
        1. YOLO annotation file (.txt) with class 0 for all products
        2. Updates product_mapping.json with original product IDs for CLIP
        
        Returns: dict with export status
        """
        import json
        from pathlib import Path
        
        if not detections:
            return {'status': 'skipped', 'reason': 'no_valid_detections'}
        
        # Get image info
        image = Image.open(image_path)
        img_w, img_h = image.size
        image_id = Path(image_path).stem
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert detections to YOLO format
        yolo_lines = []
        product_entries = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Convert xyxy to YOLO format (x_center, y_center, width, height) normalized
            x_center = ((x1 + x2) / 2) / img_w
            y_center = ((y1 + y2) / 2) / img_h
            width = (x2 - x1) / img_w
            height = (y2 - y1) / img_h
            
            # Clamp to [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))
            
            # YOLO format: class_id x_center y_center width height (class 0 for all)
            yolo_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # Product mapping for CLIP
            product_entries.append({
                'original_product_id': det['product_id'],
                'bbox': [x_center, y_center, width, height],
                'clip_conf': float(det['clip_conf']),
                'yolo_conf': float(det['yolo_conf']),
                'source': 'auto',  # 'auto' = model, 'human' = manual annotation
                'verified': False   # Human can mark as verified
            })
        
        # Save YOLO annotation
        ann_file = output_path / f"{image_id}.txt"
        with open(ann_file, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        # Update product mapping
        mapping_path = Path(mapping_file)
        product_mapping = {}
        if mapping_path.exists():
            with open(mapping_path, 'r') as f:
                product_mapping = json.load(f)
        
        # NEVER overwrite human labels
        if image_id in product_mapping:
            existing = product_mapping[image_id]
            has_human = any(p.get('source') == 'human' for p in existing)
            if has_human:
                return {'status': 'skipped', 'reason': 'has_human_labels'}
        
        product_mapping[image_id] = product_entries
        
        with open(mapping_path, 'w') as f:
            json.dump(product_mapping, f, indent=2)
        
        return {
            'status': 'exported',
            'image_id': image_id,
            'boxes': len(detections),
            'annotation_file': str(ann_file),
            'products': [e['original_product_id'] for e in product_entries]
        }
        
    def load_yolo(self):
        """Load YOLO model"""
        from ultralytics import YOLO
        print(f"Loading YOLO model: {self.yolo_model_path}")
        self.yolo_model = YOLO(self.yolo_model_path)
        
    def load_database(self, path: str = "data/product_embeddings.npz"):
        """Load product embeddings database"""
        self.product_db.load(path)
        
    def build_database(self):
        """Build product database from training data"""
        self.product_db.build_from_annotations()
        self.product_db.save()
    
    def validate_box(self, detection: dict, image_size: tuple,
                     yolo_threshold: float = 0.5,
                     clip_threshold: float = 0.7,
                     margin_threshold: float = 0.05,
                     min_area_ratio: float = 0.01,
                     max_area_ratio: float = 0.9) -> tuple:
        """
        Validate a box using gold standard rules:
        1. Detection quality: yolo_conf > threshold, reasonable area
        2. Semantic support: CLIP similarity > threshold
        3. Discriminability: top1 - top2 > margin
        
        Returns: (is_valid, rejection_reason)
        """
        img_w, img_h = image_size
        img_area = img_w * img_h
        
        x1, y1, x2, y2 = detection['bbox']
        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / img_area
        
        # Rule 1: Detection quality
        if detection['yolo_conf'] < yolo_threshold:
            return False, f"low_yolo_conf ({detection['yolo_conf']:.1%} < {yolo_threshold:.1%})"
        
        if area_ratio < min_area_ratio:
            return False, f"too_small ({area_ratio:.1%} < {min_area_ratio:.1%})"
        
        if area_ratio > max_area_ratio:
            return False, f"too_large ({area_ratio:.1%} > {max_area_ratio:.1%})"
        
        # Rule 2: Semantic support
        clip_conf = float(detection['clip_conf'])
        if clip_conf < clip_threshold:
            return False, f"low_clip_conf ({clip_conf:.1%} < {clip_threshold:.1%})"
        
        # Rule 3: Discriminability
        matches = detection['top_matches']
        if len(matches) >= 2:
            top1_score = float(matches[0][1])
            top2_score = float(matches[1][1])
            margin = top1_score - top2_score
            if margin < margin_threshold:
                return False, f"low_margin ({margin:.1%} < {margin_threshold:.1%})"
        
        return True, "valid"
    
    def compute_iou(self, box1: list, box2: list) -> float:
        """Compute IoU between two boxes [x1,y1,x2,y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0
    
    def apply_nms(self, detections: list, iou_threshold: float = 0.5) -> list:
        """Apply non-maximum suppression to remove redundant boxes"""
        if not detections:
            return []
        
        # Sort by CLIP confidence (highest first)
        sorted_dets = sorted(detections, key=lambda x: float(x['clip_conf']), reverse=True)
        
        keep = []
        for det in sorted_dets:
            # Check if this box overlaps with any kept box
            is_redundant = False
            for kept_det in keep:
                iou = self.compute_iou(det['bbox'], kept_det['bbox'])
                if iou > iou_threshold:
                    is_redundant = True
                    break
            
            if not is_redundant:
                keep.append(det)
        
        return keep
    
    def classify_image(self, image_path: str, conf_threshold: float = 0.25,
                       validate: bool = True) -> list:
        """Run full pipeline: detect products and classify them
        
        Args:
            image_path: Path to image
            conf_threshold: Initial YOLO confidence threshold
            validate: Apply gold standard box validation rules
        
        Returns list of detections with product IDs:
        [{'bbox': [x1,y1,x2,y2], 'yolo_conf': 0.95, 'product_id': '1133', 'clip_conf': 0.87}, ...]
        """
        if self.yolo_model is None:
            self.load_yolo()
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_size = image.size
        
        # Run YOLO detection
        results = self.yolo_model(image_path, verbose=False, conf=conf_threshold)
        
        detections = []
        rejected = []
        
        for r in results:
            for box in r.boxes:
                # Get box coordinates (xyxy format)
                xyxy = box.xyxy[0].tolist()
                yolo_conf = box.conf[0].item()
                
                # Crop the detection
                x1, y1, x2, y2 = [int(c) for c in xyxy]
                crop = image.crop((x1, y1, x2, y2))
                
                # Get CLIP embedding and classify
                embedding = get_image_embedding(crop)
                matches = self.product_db.classify(embedding, top_k=3)
                
                if matches:
                    product_id, clip_conf = matches[0]
                else:
                    product_id, clip_conf = "unknown", 0.0
                
                det = {
                    'bbox': xyxy,
                    'yolo_conf': yolo_conf,
                    'product_id': product_id,
                    'clip_conf': clip_conf,
                    'top_matches': matches
                }
                
                # Apply validation if requested
                if validate:
                    is_valid, reason = self.validate_box(det, image_size)
                    det['valid'] = is_valid
                    det['validation_reason'] = reason
                    
                    if is_valid:
                        detections.append(det)
                    else:
                        rejected.append(det)
                else:
                    det['valid'] = True
                    det['validation_reason'] = 'validation_skipped'
                    detections.append(det)
        
        # Rule 4: Non-redundancy (NMS)
        if validate:
            detections = self.apply_nms(detections)
        
        return detections


def main():
    """Build the product database and test classification"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CLIP Product Classifier')
    parser.add_argument('--build', action='store_true', help='Build embeddings database')
    parser.add_argument('--test', type=str, help='Test on an image')
    parser.add_argument('--auto-label', type=str, help='Auto-label image(s) and export for YOLO training')
    parser.add_argument('--export', action='store_true', help='Export validated boxes for training (use with --test)')
    args = parser.parse_args()
    
    classifier = CLIPClassifier()
    
    if args.build:
        print("Building product embeddings database...")
        load_clip_model()
        classifier.build_database()
        print("Done!")
    
    elif args.test:
        print(f"Testing on: {args.test}")
        classifier.load_database()
        
        detections = classifier.classify_image(args.test, validate=True)
        
        print(f"\n{'='*60}")
        print(f"VALID DETECTIONS: {len(detections)}")
        print(f"{'='*60}")
        print("\nBox Selection Rules Applied:")
        print("  1. YOLO conf > 50%")
        print("  2. CLIP conf > 70%")
        print("  3. Top1-Top2 margin > 5%")
        print("  4. Area: 1-90% of image")
        print("  5. NMS (IoU > 50%)")
        
        for i, det in enumerate(detections):
            print(f"\nDetection {i+1}: [VALID]")
            print(f"  YOLO confidence: {det['yolo_conf']:.1%}")
            print(f"  Product ID: {det['product_id']}")
            print(f"  CLIP confidence: {float(det['clip_conf']):.1%}")
            top3 = [(pid, f"{float(conf):.1%}") for pid, conf in det['top_matches'][:3]]
            print(f"  Top 3 matches: {top3}")
            if len(det['top_matches']) >= 2:
                margin = float(det['top_matches'][0][1]) - float(det['top_matches'][1][1])
                print(f"  Discriminability margin: {margin:.1%}")
    
    elif args.auto_label:
        from pathlib import Path
        import shutil
        
        print(f"Auto-labeling: {args.auto_label}")
        classifier.load_database()
        
        # Handle single image or directory
        target = Path(args.auto_label)
        if target.is_file():
            images = [target]
        elif target.is_dir():
            images = list(target.glob("*.jpg")) + list(target.glob("*.png"))
        else:
            print(f"Not found: {args.auto_label}")
            return
        
        # Load metadata directory
        metadata_dir = Path("data/working/metadata")
        
        def get_metadata_products(image_id):
            """Get product ASINs from metadata file"""
            meta_file = metadata_dir / f"{image_id}.json"
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                return list(meta.get('BIN_FCSKU_DATA', {}).keys())
            return []
        
        print(f"Found {len(images)} images to process")
        print(f"\n{'='*60}")
        print("AUTO-LABELING FOR YOLO TRAINING")
        print("(Using metadata for single-product images)")
        print(f"{'='*60}")
        
        stats = {'exported': 0, 'skipped': 0, 'total_boxes': 0, 'metadata_labeled': 0}
        
        # Pre-filter: only process single-product images (fast path, no CLIP)
        single_product_images = []
        multi_product_images = []
        
        for img_path in images:
            image_id = img_path.stem
            meta_products = get_metadata_products(image_id)
            if len(meta_products) == 1:
                single_product_images.append((img_path, meta_products[0]))
            elif len(meta_products) > 1:
                multi_product_images.append((img_path, meta_products))
        
        print(f"  Single-product images: {len(single_product_images)} (fast path)")
        print(f"  Multi-product images: {len(multi_product_images)} (needs CLIP)")
        print(f"  No metadata: {len(images) - len(single_product_images) - len(multi_product_images)}")
        
        # Fast path: Single-product images (YOLO only, no CLIP)
        print(f"\n[Phase 1] Processing single-product images (YOLO only)...")
        for img_path, product_id in single_product_images:
            # Run YOLO only
            if classifier.yolo_model is None:
                classifier.load_yolo()
            
            results = classifier.yolo_model(str(img_path), verbose=False, conf=0.5)
            
            high_conf_boxes = []
            for r in results:
                for box in r.boxes:
                    if box.conf[0].item() > 0.5:
                        xyxy = box.xyxy[0].tolist()
                        high_conf_boxes.append({
                            'bbox': xyxy,
                            'yolo_conf': box.conf[0].item(),
                            'product_id': product_id,
                            'clip_conf': 1.0,
                            'top_matches': [(product_id, 1.0)],
                            'source': 'metadata'
                        })
            
            # Only export if exactly 1 box found
            if len(high_conf_boxes) == 1:
                result = classifier.export_for_training(str(img_path), high_conf_boxes)
                if result['status'] == 'exported':
                    stats['exported'] += 1
                    stats['total_boxes'] += 1
                    stats['metadata_labeled'] += 1
                    print(f"  {img_path.name}: 1 box -> ['{product_id}'] (metadata)")
            else:
                stats['skipped'] += 1
        
        # Phase 2: Multi-product images (CLIP matches boxes to products)
        print(f"\n[Phase 2] Processing {len(multi_product_images)} multi-product images (CLIP matching)...")
        stats['clip_labeled'] = 0
        
        for i, (img_path, meta_products) in enumerate(multi_product_images):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(multi_product_images)} images...")
            # Run YOLO
            if classifier.yolo_model is None:
                classifier.load_yolo()
            
            results = classifier.yolo_model(str(img_path), verbose=False, conf=0.5)
            
            high_conf_boxes = []
            for r in results:
                for box in r.boxes:
                    if box.conf[0].item() > 0.5:
                        high_conf_boxes.append({
                            'bbox': box.xyxy[0].tolist(),
                            'yolo_conf': box.conf[0].item()
                        })
            
            # Only process if #boxes == #products in metadata
            if len(high_conf_boxes) != len(meta_products):
                stats['skipped'] += 1
                continue
            
            # Use CLIP to classify each box
            image = Image.open(img_path).convert('RGB')
            all_match = True
            labeled_boxes = []
            
            for box_data in high_conf_boxes:
                x1, y1, x2, y2 = [int(c) for c in box_data['bbox']]
                crop = image.crop((x1, y1, x2, y2))
                
                embedding = get_image_embedding(crop)
                matches = classifier.product_db.classify(embedding, top_k=1)
                
                if matches:
                    clip_product, clip_conf = matches[0]
                    
                    # Check if CLIP prediction is in metadata products
                    if clip_product in meta_products and float(clip_conf) > 0.7:
                        box_data['product_id'] = clip_product
                        box_data['clip_conf'] = float(clip_conf)
                        box_data['top_matches'] = matches
                        box_data['source'] = 'clip+metadata'
                        labeled_boxes.append(box_data)
                    else:
                        all_match = False
                        break
                else:
                    all_match = False
                    break
            
            # Only export if ALL boxes matched metadata products
            if all_match and len(labeled_boxes) == len(meta_products):
                result = classifier.export_for_training(str(img_path), labeled_boxes)
                if result['status'] == 'exported':
                    stats['exported'] += 1
                    stats['total_boxes'] += result['boxes']
                    stats['clip_labeled'] += 1
                    print(f"  {img_path.name}: {result['boxes']} boxes -> {result['products']} (CLIP+metadata)")
            else:
                stats['skipped'] += 1
        
        print(f"\n{'='*60}")
        print(f"SUMMARY:")
        print(f"  Images exported: {stats['exported']}")
        print(f"    - From metadata (1 product): {stats['metadata_labeled']}")
        print(f"    - From CLIP+metadata (multi): {stats.get('clip_labeled', 0)}")
        print(f"  Images skipped: {stats['skipped']}")
        print(f"  Total boxes: {stats['total_boxes']}")
        print(f"\nAnnotations saved to: data/working/annotations/")
        print(f"Product mapping updated: data/product_mapping.json")
        print(f"\nNext: Run 'python create_snapshot.py' then deploy to Kaggle")
    
    else:
        print("Usage:")
        print("  Build database:    python clip_classifier.py --build")
        print("  Test on image:     python clip_classifier.py --test path/to/image.jpg")
        print("  Auto-label:        python clip_classifier.py --auto-label path/to/images/")
        print("  Auto-label single: python clip_classifier.py --auto-label path/to/image.jpg")


if __name__ == "__main__":
    main()
