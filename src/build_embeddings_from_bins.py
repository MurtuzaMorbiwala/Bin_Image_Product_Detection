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

# Hardcoded file locations
CLIP_MODEL_PATH = r"C:\Users\bindi\.cache\clip\ViT-B-32.pt"
IMAGES_DIR = r"data\working\images"
METADATA_DIR = r"data\working\metadata"
YOLO_MODEL_PATH = r"kaggle_output\best.pt"
EMBEDDINGS_OUTPUT = r"data\product_embeddings.npz"


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


def load_clip_model(model_path: str = CLIP_MODEL_PATH):
    """Load CLIP model from local path"""
    global clip_model, clip_preprocess, device
    
    ensure_clip_installed()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model from '{model_path}' on {device}...")
    
    clip_model, clip_preprocess = clip.load(model_path, device=device)
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


def load_ground_truth(metadata_file: Path) -> dict:
    """Load ground truth from bin metadata JSON."""
    try:
        with open(metadata_file) as f:
            metadata = json.load(f)
        products = {}
        if 'BIN_FCSKU_DATA' in metadata:
            for product_id, product_data in metadata['BIN_FCSKU_DATA'].items():
                products[product_id] = {
                    'id': product_id,
                    'name': product_data.get('name', product_data.get('normalizedName', 'Unknown')),
                }
        return products
    except Exception:
        return {}


class ProductDatabase:
    """Database of product embeddings for classification"""
    
    def __init__(self, embeddings_path: Optional[str] = None):
        self.embeddings = {}
        self.embeddings_path = embeddings_path or EMBEDDINGS_OUTPUT
        
    def add_product_embedding(self, product_id: str, embedding: np.ndarray):
        """Add an embedding for a product"""
        if product_id not in self.embeddings:
            self.embeddings[product_id] = []
        self.embeddings[product_id].append(embedding)
    
    def build_from_ground_truth(self,
                                images_dir: str = IMAGES_DIR,
                                metadata_dir: str = METADATA_DIR,
                                yolo_model_path: str = YOLO_MODEL_PATH,
                                min_occurrences: int = 3):
        """
        Build embeddings using YOLO detections + ground truth metadata
        This is the weakly supervised approach from the first script
        """
        from ultralytics import YOLO
        from collections import defaultdict
        from tqdm import tqdm
        
        images_path = Path(images_dir)
        metadata_path = Path(metadata_dir)
        
        print("\n" + "="*60)
        print("BUILDING CLIP DATABASE FROM GROUND TRUTH")
        print("="*60)
        
        # Load YOLO model
        print(f"\nLoading YOLO model: {yolo_model_path}")
        yolo = YOLO(yolo_model_path)
        
        product_crops = defaultdict(list)
        product_names = {}
        product_bin_count = defaultdict(int)
        image_files = sorted(list(images_path.glob("*.jpg")))
        
        print(f"\nProcessing {len(image_files)} images...")
        
        # Extract crops using weakly supervised association
        for img_file in tqdm(image_files, desc="Scanning Bins"):
            image_id = img_file.stem
            metadata_file = metadata_path / f"{image_id}.json"
            
            ground_truth = load_ground_truth(metadata_file)
            if not ground_truth:
                continue
            
            # Track product occurrence
            for pid, pdata in ground_truth.items():
                product_names[pid] = pdata['name']
                product_bin_count[pid] += 1
                
            # Run YOLO to find all "objects" in the bin
            img = Image.open(img_file).convert('RGB')
            results = yolo(img, conf=0.25, verbose=False)
            if not results or len(results[0].boxes) == 0:
                continue
            
            # Associate EVERY crop in this bin with EVERY product in the metadata
            for box in results[0].boxes.xyxy.cpu().numpy():
                crop = img.crop((box[0], box[1], box[2], box[3]))
                for pid in ground_truth.keys():
                    product_crops[pid].append(crop)
        
        # Generate consensus embeddings
        print(f"\nEncoding {len(product_crops)} products...")
        total_crops = 0
        
        for pid, crops in tqdm(product_crops.items(), desc="Generating Embeddings"):
            if product_bin_count[pid] < min_occurrences:
                continue
            
            # Limit crops per product to 50 to maintain speed and memory
            sample_crops = crops[:50]
            batch_vecs = []
            
            for crop in sample_crops:
                embedding = get_image_embedding(crop)
                batch_vecs.append(embedding)
                total_crops += 1
            
            if batch_vecs:
                # Consensus: The mean vector of all crops from all bins containing this product
                avg_vec = np.mean(batch_vecs, axis=0)
                avg_vec /= np.linalg.norm(avg_vec)  # Re-normalize the average
                self.add_product_embedding(pid, avg_vec)
        
        print(f"\n{'='*60}")
        print(f"FINAL SUMMARY:")
        print(f"  Total unique products: {len(self.embeddings)}")
        print(f"  Total crops processed: {total_crops}")
        print(f"  Products with min {min_occurrences} occurrences: {len(self.embeddings)}")
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
    
    def __init__(self, yolo_model_path: str = YOLO_MODEL_PATH):
        self.yolo_model_path = yolo_model_path
        self.yolo_model = None
        self.product_db = ProductDatabase()
    
    def load_yolo(self):
        """Load YOLO model"""
        from ultralytics import YOLO
        print(f"Loading YOLO model: {self.yolo_model_path}")
        self.yolo_model = YOLO(self.yolo_model_path)
        
    def load_database(self, path: str = EMBEDDINGS_OUTPUT):
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
    
    parser = argparse.ArgumentParser(description='CLIP Product Classifier')
    parser.add_argument('--build', action='store_true', help='Build CLIP database')
    parser.add_argument('--test', type=str, help='Test on an image')
    parser.add_argument('--yolo-model', type=str, default=YOLO_MODEL_PATH,
                        help='Path to YOLO model')
    
    args = parser.parse_args()
    
    if args.build:
        print("Building CLIP database from ground truth...")
        load_clip_model()
        
        db = ProductDatabase()
        db.build_from_ground_truth()
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
        print("  Build database:  python script.py --build")
        print("  Test on image:   python script.py --test image.jpg")


if __name__ == "__main__":
    main()