"""Simple CLIP classifier - only uses crops that match ground truth"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Dict, List

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
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.cpu().numpy().flatten()


def crop_from_yolo_box(image: Image.Image, bbox: list) -> Image.Image:
    """Crop image using YOLO format bbox [x_center, y_center, width, height]"""
    width, height = image.size
    
    x_center, y_center, box_w, box_h = bbox
    
    x1 = int((x_center - box_w/2) * width)
    y1 = int((y_center - box_h/2) * height)
    x2 = int((x_center + box_w/2) * width)
    y2 = int((y_center + box_h/2) * height)
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    
    return image.crop((x1, y1, x2, y2))


def filter_clustering_data(cluster_data: Dict, metadata_dir: str = "data/working/metadata") -> Dict:
    """
    Filter clustering output to only keep good quality crops:
    1. Remove duplicate boxes (keep highest confidence)
    2. Only keep YOLO confidence > 0.3
    3. Only keep crops where predicted product ID matches ground truth metadata
    """
    metadata_path = Path(metadata_dir)
    filtered = {}
    
    stats = {
        'total_images': len(cluster_data),
        'total_crops': sum(len(products) for products in cluster_data.values()),
        'kept_images': 0,
        'kept_crops': 0,
        'removed_low_conf': 0,
        'removed_no_metadata': 0,
        'removed_wrong_product': 0,
    }
    
    for image_id, products in cluster_data.items():
        # 1. Deduplicate boxes (keep highest confidence)
        unique_boxes = {}
        for p in products:
            key = tuple(p['bbox'])
            if key not in unique_boxes or p.get('yolo_conf', 0) > unique_boxes[key].get('yolo_conf', 0):
                unique_boxes[key] = p
        products = list(unique_boxes.values())
        
        # 2. Filter by confidence
        high_conf_products = [p for p in products if p.get('yolo_conf', 0) >= 0.3]
        stats['removed_low_conf'] += len(products) - len(high_conf_products)
        
        if not high_conf_products:
            continue
        
        # 3. Load ground truth metadata
        meta_file = metadata_path / f"{image_id}.json"
        if not meta_file.exists():
            stats['removed_no_metadata'] += len(high_conf_products)
            continue
        
        with open(meta_file, 'r') as f:
            meta = json.load(f)
        
        ground_truth_ids = set(meta.get('BIN_FCSKU_DATA', {}).keys())
        
        # 4. Only keep products that match ground truth
        validated_products = []
        for p in high_conf_products:
            if p['original_product_id'] in ground_truth_ids:
                validated_products.append(p)
            else:
                stats['removed_wrong_product'] += 1
        
        if validated_products:
            filtered[image_id] = validated_products
            stats['kept_images'] += 1
            stats['kept_crops'] += len(validated_products)
    
    print(f"\n{'='*60}")
    print(f"FILTERING RESULTS:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Total crops: {stats['total_crops']}")
    print(f"")
    print(f"  ✓ Kept images: {stats['kept_images']}")
    print(f"  ✓ Kept crops: {stats['kept_crops']}")
    print(f"")
    print(f"  Removed:")
    print(f"    - Low confidence (<0.3): {stats['removed_low_conf']}")
    print(f"    - No metadata file: {stats['removed_no_metadata']}")
    print(f"    - Wrong product ID: {stats['removed_wrong_product']}")
    print(f"{'='*60}")
    
    return filtered


class ProductDatabase:
    """Database of product embeddings for classification"""
    
    def __init__(self, embeddings_path: Optional[str] = None):
        self.embeddings = {}
        self.embeddings_path = embeddings_path or "data/product_embeddings.npz"
        
    def add_product_embedding(self, product_id: str, embedding: np.ndarray):
        """Add an embedding for a product"""
        if product_id not in self.embeddings:
            self.embeddings[product_id] = []
        self.embeddings[product_id].append(embedding)
    
    def build_from_sources(self,
                          manual_file: str = "data/product_mapping.json",
                          cluster_file: str = "clustered_output/product_mapping.json",
                          images_dir: str = "data/snapshots/binsense/images"):
        """
        Build embeddings from manual annotations and clustering data
        Only uses crops that match ground truth
        """
        images_path = Path(images_dir)
        total_crops = 0
        
        print("\n" + "="*60)
        print("BUILDING CLIP DATABASE")
        print("="*60)
        
        # Load manual annotations
        if Path(manual_file).exists():
            print(f"\n✓ Loading manual annotations: {manual_file}")
            with open(manual_file, 'r') as f:
                manual_data = json.load(f)
            
            manual_crops = 0
            for image_id, products in manual_data.items():
                image_file = images_path / f"{image_id}.jpg"
                if not image_file.exists():
                    continue
                
                image = Image.open(image_file).convert('RGB')
                
                for product in products:
                    crop = crop_from_yolo_box(image, product['bbox'])
                    embedding = get_image_embedding(crop)
                    self.add_product_embedding(product['original_product_id'], embedding)
                    manual_crops += 1
            
            print(f"  Added {manual_crops} crops from manual annotations")
            total_crops += manual_crops
        else:
            print(f"\n⚠️  Manual annotations not found: {manual_file}")
        
        # Load clustering data
        if Path(cluster_file).exists():
            print(f"\n✓ Loading clustering output: {cluster_file}")
            with open(cluster_file, 'r') as f:
                cluster_data = json.load(f)
            
            # Filter to only validated crops
            cluster_data = filter_clustering_data(cluster_data)
            
            cluster_crops = 0
            for image_id, products in cluster_data.items():
                image_file = images_path / f"{image_id}.jpg"
                if not image_file.exists():
                    continue
                
                image = Image.open(image_file).convert('RGB')
                
                for product in products:
                    crop = crop_from_yolo_box(image, product['bbox'])
                    embedding = get_image_embedding(crop)
                    self.add_product_embedding(product['original_product_id'], embedding)
                    cluster_crops += 1
            
            print(f"  Added {cluster_crops} crops from clustering")
            total_crops += cluster_crops
        else:
            print(f"\n⚠️  Clustering output not found: {cluster_file}")
        
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY:")
        print(f"  Total unique products: {len(self.embeddings)}")
        print(f"  Total crops/embeddings: {total_crops}")
        print(f"  Avg crops per product: {total_crops/max(1, len(self.embeddings)):.1f}")
        print(f"{'='*60}")
        
        return self
    
    def save(self, path: Optional[str] = None):
        """Save embeddings to file"""
        path = path or self.embeddings_path
        
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_data = {}
        for product_id, emb_list in self.embeddings.items():
            save_data[f"product_{product_id}"] = np.array(emb_list)
        
        np.savez(path, **save_data)
        print(f"\n✓ Saved embeddings to {path}")
    
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
        """Classify an embedding against the database"""
        if not self.embeddings:
            return []
        
        scores = {}
        for product_id, emb_list in self.embeddings.items():
            similarities = [np.dot(embedding, ref_emb) for ref_emb in emb_list]
            scores[product_id] = max(similarities)
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:top_k]


class CLIPClassifier:
    """Main classifier combining YOLO detection with CLIP classification"""
    
    def __init__(self, yolo_model_path: str = "kaggle_output/best.pt"):
        self.yolo_model_path = yolo_model_path
        self.yolo_model = None
        self.product_db = ProductDatabase()
    
    def load_yolo(self):
        """Load YOLO model"""
        from ultralytics import YOLO
        print(f"Loading YOLO model: {self.yolo_model_path}")
        self.yolo_model = YOLO(self.yolo_model_path)
        
    def load_database(self, path: str = "data/product_embeddings.npz"):
        """Load product embeddings database"""
        self.product_db.load(path)
    
    def classify_image(self, image_path: str, conf_threshold: float = 0.25) -> list:
        """
        Run full pipeline: detect products and classify them
        
        Returns list of detections:
        [{
            'bbox': [x1,y1,x2,y2],
            'yolo_conf': 0.95,
            'product_id': 'B00E9J3MLM',
            'clip_conf': 0.87,
            'top_matches': [('B00E9J3MLM', 0.87), ...]
        }, ...]
        """
        if self.yolo_model is None:
            self.load_yolo()
        
        image = Image.open(image_path).convert('RGB')
        results = self.yolo_model(image_path, verbose=False, conf=conf_threshold)
        
        detections = []
        
        for r in results:
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                yolo_conf = box.conf[0].item()
                
                x1, y1, x2, y2 = [int(c) for c in xyxy]
                crop = image.crop((x1, y1, x2, y2))
                
                embedding = get_image_embedding(crop)
                matches = self.product_db.classify(embedding, top_k=3)
                
                if matches:
                    product_id, clip_conf = matches[0]
                else:
                    product_id, clip_conf = "unknown", 0.0
                
                detections.append({
                    'bbox': xyxy,
                    'yolo_conf': yolo_conf,
                    'product_id': product_id,
                    'clip_conf': clip_conf,
                    'top_matches': matches,
                    'crop': crop  # Include crop for visualization
                })
        
        return detections


def main():
    """Build database or test classification"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CLIP Product Classifier (Simple)')
    parser.add_argument('--build', action='store_true', help='Build CLIP database')
    parser.add_argument('--test', type=str, help='Test on an image')
    parser.add_argument('--yolo-model', type=str, default='kaggle_output/best.pt',
                        help='Path to YOLO model')
    
    args = parser.parse_args()
    
    if args.build:
        print("Building CLIP database...")
        print("(Only using crops that match ground truth)")
        load_clip_model()
        
        db = ProductDatabase()
        db.build_from_sources()
        db.save()
        print("\n✓ Done!")
    
    elif args.test:
        classifier = CLIPClassifier(yolo_model_path=args.yolo_model)
        classifier.load_database()
        
        detections = classifier.classify_image(args.test)
        
        print(f"\n{'='*60}")
        print(f"DETECTIONS: {len(detections)}")
        print(f"{'='*60}")
        
        for i, det in enumerate(detections):
            print(f"\nDetection {i+1}:")
            print(f"  Product ID: {det['product_id']}")
            print(f"  YOLO conf: {det['yolo_conf']:.1%}")
            print(f"  CLIP conf: {float(det['clip_conf']):.1%}")
            print(f"  Top 3: {[(p, f'{float(c):.1%}') for p, c in det['top_matches'][:3]]}")
    
    else:
        print("\nUsage:")
        print("  Build database:  python -m src.clip_classifier_simple --build")
        print("  Test on image:   python -m src.clip_classifier_simple --test image.jpg")


if __name__ == "__main__":
    main()