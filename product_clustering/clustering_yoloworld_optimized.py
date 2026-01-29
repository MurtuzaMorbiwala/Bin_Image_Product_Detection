"""
BinSense Product-by-Product Clustering - SIMPLE EXCLUSIVE APPROACH

THE SIMPLEST SOLUTION:
1. Keep only boxes ≤75% of image (no full-image detections)
2. Find the most similar cluster for each product
3. Validate: Cluster must appear ONLY in images containing this product
4. That's it!

This uses your metadata for validation, not complex constraints.
Simple, fast, and effective.

Requirements:
    pip install torch torchvision
    pip install ultralytics>=8.1.0
    pip install Pillow numpy
    pip install git+https://github.com/openai/CLIP.git
    pip install scikit-learn
"""

import sys
import os
from pathlib import Path
import json
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLOWorld
import clip
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# ============================================================
# CONFIGURATION - HARDCODED PATHS
# ============================================================

DATA_DIR = Path("C:/Users/bindi/PythonDev/BinSense2/data/working")
IMAGES_DIR = Path("C:/Users/bindi/PythonDev/BinSense2/data/working/images")
METADATA_DIR = Path("C:/Users/bindi/PythonDev/BinSense2/data/working/metadata")
OUTPUT_DIR = Path("C:/Users/bindi/PythonDev/BinSense2/clustered_output")
CACHE_DIR = OUTPUT_DIR / "cache"

# ============================================================
# SIMPLE PARAMETERS
# ============================================================

# Box filtering - ONLY size-based
MAX_BOX_AREA = 0.75          # Only boxes ≤75% of image
MIN_BOX_AREA = 0.01          # Ignore tiny boxes
MIN_YOLO_CONF = 0.05         # Basic confidence floor

# Clustering - Simple similarity
SIMILARITY_THRESHOLD = 0.85   # Boxes must be at least 85% similar
MIN_CLUSTER_SIZE = 1          # Accept even single boxes if exclusive

# Exclusivity - The key filter!
REQUIRE_EXCLUSIVITY = True    # Cluster must ONLY appear in images with this product
MAX_CONTAMINATION = 0.0       # 0% tolerance for other products (strict)

# Other
CLIP_MODEL = "ViT-B/32"
YOLO_WORLD_MODEL = "yolov8x-worldv2.pt"
DETECTION_CLASSES = ["product", "box", "package", "item", "object", "container"]
YOLO_CONF = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# DISPLAY
# ============================================================
print("="*60)
print("BINSENSE CLUSTERING - SIMPLE EXCLUSIVE APPROACH")
print("="*60)
print("\nRunning on:", "GPU" if torch.cuda.is_available() else "CPU")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"\n🎯 SIMPLE APPROACH:")
print(f"  1. Keep boxes ≤{MAX_BOX_AREA*100:.0f}% of image")
print(f"  2. Find most similar cluster (≥{SIMILARITY_THRESHOLD})")
print(f"  3. Validate: Cluster ONLY in images with this product")
print(f"  4. Done!")
print(f"\n📊 Parameters:")
print(f"  - Max box area: {MAX_BOX_AREA*100:.0f}%")
print(f"  - Similarity threshold: {SIMILARITY_THRESHOLD}")
print(f"  - Exclusivity required: {REQUIRE_EXCLUSIVITY}")
print(f"  - Max contamination: {MAX_CONTAMINATION*100:.0f}%")

# ============================================================
# SETUP
# ============================================================
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

# Verify paths
for path, name in [(DATA_DIR, "Data"), (IMAGES_DIR, "Images"), (METADATA_DIR, "Metadata")]:
    if not path.exists():
        print(f"\n[!] ERROR: {name} directory not found at: {path}")
        sys.exit(1)

# Load models
print(f"\n[*] Loading YOLO-World model: {YOLO_WORLD_MODEL}")
try:
    yolo_model = YOLOWorld(YOLO_WORLD_MODEL)
    yolo_model.set_classes(DETECTION_CLASSES)
    print(f"[+] YOLO-World loaded successfully")
except Exception as e:
    print(f"\n[!] ERROR loading YOLO-World: {e}")
    sys.exit(1)

print(f"[*] Loading CLIP model: {CLIP_MODEL}")
clip_model, preprocess = clip.load(CLIP_MODEL, device=device)
print("[+] Models loaded successfully")

# Load existing mapping
mapping_file = OUTPUT_DIR / "product_mapping.json"
existing_mapping = json.load(open(mapping_file)) if mapping_file.exists() else {}

known_products = set()
for products in existing_mapping.values():
    for p in products:
        pid = p.get("original_product_id", "")
        if pid.startswith("B"):
            known_products.add(pid)

print(f"[+] Already know {len(known_products)} products")

# ============================================================
# BUILD PRODUCT INDEX + IMAGE INDEX
# ============================================================
print("\n[*] Scanning metadata files...")
product_to_images = defaultdict(list)
image_to_products = {}
all_image_ids = set()

for meta_file in METADATA_DIR.glob("*.json"):
    img_id = meta_file.stem
    all_image_ids.add(img_id)
    try:
        meta = json.load(open(meta_file))
        products_in_bin = list(meta.get("BIN_FCSKU_DATA", {}).keys())
        
        image_products = []
        for product_id in products_in_bin:
            if product_id.startswith("B"):
                product_to_images[product_id].append(img_id)
                image_products.append(product_id)
        
        image_to_products[img_id] = image_products
        
    except Exception as e:
        continue

total_products = len(product_to_images)
unknown_products = [p for p in product_to_images.keys() if p not in known_products]

print(f"[+] Total unique products: {total_products}")
print(f"[+] Total unique images: {len(all_image_ids)}")
print(f"[+] Already known: {len(known_products)}")
print(f"[+] Unknown (to learn): {len(unknown_products)}")

# ============================================================
# PHASE 1+2: YOLO DETECTION + CLIP EXTRACTION (REUSE CACHE)
# ============================================================
print("\n" + "="*60)
print("PHASE 1+2: YOLO DETECTION + CLIP EXTRACTION")
print("="*60)

yolo_cache_file = CACHE_DIR / "yolo_detections.json"
clip_cache_file = CACHE_DIR / "clip_embeddings_cache.json"

# Load caches
if yolo_cache_file.exists():
    print(f"[*] Loading cached YOLO detections...")
    with open(yolo_cache_file, 'r') as f:
        yolo_cache = json.load(f)
    print(f"[+] Loaded {len(yolo_cache)} cached YOLO detections")
else:
    yolo_cache = {}

if clip_cache_file.exists():
    print(f"[*] Loading cached CLIP embeddings...")
    with open(clip_cache_file, 'r') as f:
        clip_cache = json.load(f)
    print(f"[+] Loaded {len(clip_cache)} cached CLIP embeddings")
else:
    clip_cache = {}

images_to_process = [img_id for img_id in all_image_ids if img_id not in clip_cache]

if not images_to_process:
    print(f"[+] All {len(all_image_ids)} images already processed!")
else:
    print(f"\n[*] Processing {len(images_to_process)} images (YOLO + CLIP)...")
    BATCH_SIZE = 100
    
    for batch_start in range(0, len(images_to_process), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(images_to_process))
        batch_images = images_to_process[batch_start:batch_end]
        
        print(f"\n[*] Processing batch: images {batch_start + 1}-{batch_end}/{len(images_to_process)}")
        
        for img_id in tqdm(batch_images, desc=f"Batch {batch_start//BATCH_SIZE + 1}"):
            img_path = IMAGES_DIR / f"{img_id}.jpg"
            
            if not img_path.exists():
                yolo_cache[img_id] = []
                clip_cache[img_id] = []
                continue
            
            # YOLO detection
            try:
                results = yolo_model(str(img_path), verbose=False, conf=YOLO_CONF)
                boxes = [{"bbox": b.xyxy[0].tolist(), "conf": b.conf[0].item()} 
                        for r in results for b in r.boxes]
                yolo_cache[img_id] = boxes
            except Exception as e:
                tqdm.write(f"[!] YOLO error on {img_id}: {e}")
                yolo_cache[img_id] = []
                clip_cache[img_id] = []
                continue
            
            # CLIP extraction
            if not boxes:
                clip_cache[img_id] = []
                continue
            
            try:
                image = Image.open(img_path).convert("RGB")
                w, h = image.size
                
                box_embeddings = []
                
                for box_idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(c) for c in box["bbox"]]
                    
                    if x2 <= x1 or y2 <= y1 or (x2-x1) < 20 or (y2-y1) < 20:
                        continue
                    
                    crop = image.crop((max(0,x1), max(0,y1), min(w,x2), min(h,y2)))
                    
                    with torch.no_grad():
                        emb = clip_model.encode_image(preprocess(crop).unsqueeze(0).to(device))
                        emb = (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy().flatten()
                    
                    box_embeddings.append({
                        'box_idx': box_idx,
                        'embedding': emb.tolist(),
                        'bbox': box['bbox'],
                        'conf': box['conf'],
                        'bbox_norm': [
                            ((x1+x2)/2)/w, ((y1+y2)/2)/h,
                            (x2-x1)/w, (y2-y1)/h
                        ]
                    })
                
                clip_cache[img_id] = box_embeddings
                
            except Exception as e:
                tqdm.write(f"[!] CLIP error on {img_id}: {e}")
                clip_cache[img_id] = []
        
        # Save caches
        print(f"[*] Saving caches (batch {batch_start//BATCH_SIZE + 1})...")
        with open(yolo_cache_file, 'w') as f:
            json.dump(yolo_cache, f, indent=2)
        with open(clip_cache_file, 'w') as f:
            json.dump(clip_cache, f, indent=2)
        print(f"[+] Saved! Processed {batch_end}/{len(images_to_process)} images total")
    
    print(f"\n[+] All images processed and cached!")

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings"""
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


def find_exclusive_cluster_simple(product_id, candidates, image_to_products):
    """
    Find the largest cluster of similar boxes that ONLY appears in images
    containing this product (exclusivity check).
    
    Returns: (cluster_indices, exclusivity_score, contamination_images)
    """
    n = len(candidates)
    if n == 0:
        return [], 0.0, []
    
    # Calculate all pairwise similarities
    embeddings = [c['embedding'] for c in candidates]
    similarities = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            if i == j:
                similarities[i][j] = 1.0
            else:
                sim = cosine_similarity(embeddings[i], embeddings[j])
                similarities[i][j] = sim
                similarities[j][i] = sim
    
    # Find all possible clusters (greedy approach - start from each box)
    all_clusters = []
    
    for start_idx in range(n):
        cluster = [start_idx]
        
        # Add all boxes similar to ALL current cluster members
        for candidate_idx in range(n):
            if candidate_idx in cluster:
                continue
            
            # Check if similar to ALL cluster members
            all_similar = all(
                similarities[candidate_idx, cluster_member] >= SIMILARITY_THRESHOLD
                for cluster_member in cluster
            )
            
            if all_similar:
                cluster.append(candidate_idx)
        
        if len(cluster) >= MIN_CLUSTER_SIZE:
            all_clusters.append(cluster)
    
    if not all_clusters:
        return [], 0.0, []
    
    # Remove duplicate clusters
    unique_clusters = []
    seen = set()
    for cluster in all_clusters:
        cluster_tuple = tuple(sorted(cluster))
        if cluster_tuple not in seen:
            seen.add(cluster_tuple)
            unique_clusters.append(cluster)
    
    # Score each cluster by EXCLUSIVITY
    scored_clusters = []
    
    for cluster_indices in unique_clusters:
        # Get all images this cluster appears in
        cluster_images = set([candidates[i]['image_id'] for i in cluster_indices])
        
        # Check exclusivity: Do ALL these images contain our product?
        images_with_product = 0
        images_without_product = 0
        contamination_images = []
        
        for img_id in cluster_images:
            products_in_image = image_to_products.get(img_id, [])
            
            if product_id in products_in_image:
                images_with_product += 1
            else:
                images_without_product += 1
                contamination_images.append(img_id)
        
        # Exclusivity score: What % of cluster images actually contain this product?
        total_images = len(cluster_images)
        exclusivity_score = images_with_product / total_images if total_images > 0 else 0
        contamination_ratio = images_without_product / total_images if total_images > 0 else 0
        
        # Calculate average similarity within cluster
        if len(cluster_indices) == 1:
            avg_similarity = 1.0
        else:
            sims = [similarities[i][j] for i in cluster_indices for j in cluster_indices if i < j]
            avg_similarity = np.mean(sims) if sims else 0
        
        scored_clusters.append({
            'indices': cluster_indices,
            'size': len(cluster_indices),
            'exclusivity_score': exclusivity_score,
            'contamination_ratio': contamination_ratio,
            'contamination_images': contamination_images,
            'avg_similarity': avg_similarity,
            'num_images': total_images,
            'composite_score': exclusivity_score * avg_similarity * np.sqrt(len(cluster_indices))
        })
    
    # Sort by exclusivity first, then size
    scored_clusters.sort(key=lambda x: (x['exclusivity_score'], x['size'], x['avg_similarity']), reverse=True)
    
    # Return best cluster if it meets exclusivity requirement
    best = scored_clusters[0]
    
    if REQUIRE_EXCLUSIVITY and best['contamination_ratio'] > MAX_CONTAMINATION:
        return [], best['exclusivity_score'], best['contamination_images']
    
    return best['indices'], best['exclusivity_score'], best['contamination_images']


# ============================================================
# PHASE 3: SIMPLE EXCLUSIVE CLUSTERING
# ============================================================
print("\n" + "="*60)
print("PHASE 3: SIMPLE EXCLUSIVE CLUSTERING")
print("="*60)
print(f"\n{'Product':<15} {'Images':<8} {'Boxes':<8} {'Valid':<8} {'Cluster':<8} {'Exclusivity':>15} {'Status':<20}")
print("-"*95)

learned_products = {}
new_labels = defaultdict(list)

stats = {
    'processed': 0,
    'learned': 0,
    'skipped_no_boxes': 0,
    'skipped_too_large': 0,
    'skipped_too_small': 0,
    'skipped_low_conf': 0,
    'skipped_no_cluster': 0,
    'skipped_contaminated': 0,
    'perfect_exclusivity': 0,  # 100% exclusive
    'high_exclusivity': 0,      # >90%
    'medium_exclusivity': 0     # 80-90%
}

# Prepare log
log_file = OUTPUT_DIR / "exclusive_clustering_log.jsonl"
log = open(log_file, 'w')

try:
    for idx, product_id in enumerate(unknown_products):
        stats['processed'] += 1
        
        if (idx + 1) % 100 == 0:
            print(f"\n[*] Progress: {idx + 1}/{len(unknown_products)} processed, {stats['learned']} learned\n")
        
        # Get images containing this product
        image_ids = product_to_images[product_id]
        
        # Collect valid boxes (simple size filtering)
        all_boxes = []
        valid_boxes = []
        
        for img_id in image_ids:
            box_embeddings = clip_cache.get(img_id, [])
            
            for box_emb in box_embeddings:
                all_boxes.append(box_emb)
                
                # Simple filters
                bbox_norm = box_emb['bbox_norm']
                yolo_conf = box_emb['conf']
                
                cx, cy, w, h = bbox_norm
                area = w * h
                
                # Filter 1: YOLO confidence
                if yolo_conf < MIN_YOLO_CONF:
                    stats['skipped_low_conf'] += 1
                    continue
                
                # Filter 2: Box too large (likely full-image)
                if area > MAX_BOX_AREA:
                    stats['skipped_too_large'] += 1
                    continue
                
                # Filter 3: Box too small
                if area < MIN_BOX_AREA:
                    stats['skipped_too_small'] += 1
                    continue
                
                # Valid box!
                valid_boxes.append({
                    'embedding': box_emb['embedding'],
                    'image_id': img_id,
                    'bbox_xyxy': box_emb['bbox'],
                    'yolo_conf': yolo_conf,
                    'bbox_norm': bbox_norm
                })
        
        if not valid_boxes:
            stats['skipped_no_boxes'] += 1
            print(f"{product_id:<15} {len(image_ids):<8} {len(all_boxes):<8} {len(valid_boxes):<8} {0:<8} {'N/A':>15} {'NoValidBoxes':<20}")
            continue
        
        # Find exclusive cluster
        cluster_indices, exclusivity_score, contamination = find_exclusive_cluster_simple(
            product_id, valid_boxes, image_to_products
        )
        
        if not cluster_indices:
            stats['skipped_no_cluster'] += 1
            if contamination:
                stats['skipped_contaminated'] += 1
                status = f"Contaminated({len(contamination)})"
            else:
                status = "NoCluster"
            print(f"{product_id:<15} {len(image_ids):<8} {len(all_boxes):<8} {len(valid_boxes):<8} {0:<8} {exclusivity_score:>15.2f} {status:<20}")
            continue
        
        # SUCCESS!
        stats['learned'] += 1
        
        # Track exclusivity
        if exclusivity_score >= 1.0:
            stats['perfect_exclusivity'] += 1
        elif exclusivity_score >= 0.9:
            stats['high_exclusivity'] += 1
        elif exclusivity_score >= 0.8:
            stats['medium_exclusivity'] += 1
        
        cluster_boxes = [valid_boxes[i] for i in cluster_indices]
        
        # Average embedding
        consensus_embedding = np.mean([b['embedding'] for b in cluster_boxes], axis=0)
        learned_products[product_id] = consensus_embedding.tolist()
        
        # Create labels
        for box in cluster_boxes:
            new_labels[box['image_id']].append({
                'original_product_id': product_id,
                'bbox': box['bbox_norm'],
                'yolo_conf': box['yolo_conf'],
                'source': 'auto_clustering_exclusive',
                'cluster_size': len(cluster_indices),
                'exclusivity_score': exclusivity_score
            })
        
        # Log details
        log_entry = {
            'product_id': product_id,
            'cluster_size': len(cluster_indices),
            'exclusivity_score': exclusivity_score,
            'num_images': len(set([b['image_id'] for b in cluster_boxes])),
            'contamination_images': contamination
        }
        log.write(json.dumps(log_entry) + '\n')
        
        # Print success
        status = "✓ Perfect" if exclusivity_score >= 1.0 else f"✓ {exclusivity_score*100:.0f}%"
        print(f"{product_id:<15} {len(image_ids):<8} {len(all_boxes):<8} {len(valid_boxes):<8} {len(cluster_indices):<8} {exclusivity_score:>15.2f} {status:<20}")

except KeyboardInterrupt:
    print("\n\nINTERRUPTED")
except Exception as e:
    print(f"\n\nERROR: {e}")
    import traceback
    traceback.print_exc()
finally:
    log.close()

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Update mapping
final_mapping = existing_mapping.copy()
for img_id, products in new_labels.items():
    if img_id in final_mapping:
        final_mapping[img_id].extend(products)
    else:
        final_mapping[img_id] = products

# Save files
with open(OUTPUT_DIR / "product_mapping.json", 'w') as f:
    json.dump(final_mapping, f, indent=2)
print(f"[+] Saved: product_mapping.json")

with open(OUTPUT_DIR / "clip_embeddings.json", 'w') as f:
    json.dump(learned_products, f, indent=2)
print(f"[+] Saved: clip_embeddings.json ({len(learned_products)} products)")

labels_dir = OUTPUT_DIR / "labels"
labels_dir.mkdir(exist_ok=True)
for img_id, products in new_labels.items():
    with open(labels_dir / f"{img_id}.txt", 'w') as f:
        for prod in products:
            bbox = prod['bbox']
            f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
print(f"[+] Saved: {len(new_labels)} label files")

# Summary
summary = {
    'starting_products': len(known_products),
    'products_processed': stats['processed'],
    'products_learned': stats['learned'],
    'success_rate': stats['learned']/stats['processed'] if stats['processed'] > 0 else 0,
    'statistics': stats,
    'config': {
        'max_box_area': MAX_BOX_AREA,
        'min_box_area': MIN_BOX_AREA,
        'similarity_threshold': SIMILARITY_THRESHOLD,
        'require_exclusivity': REQUIRE_EXCLUSIVITY,
        'max_contamination': MAX_CONTAMINATION,
        'approach': 'Simple exclusive clustering - boxes ≤75% image, high similarity, product exclusivity'
    },
    'timestamp': datetime.now().isoformat()
}

with open(OUTPUT_DIR / "clustering_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)
print(f"[+] Saved: clustering_summary.json")

# ============================================================
# FINAL STATISTICS
# ============================================================
print("\n" + "="*60)
print("FINAL STATISTICS - SIMPLE EXCLUSIVE CLUSTERING")
print("="*60)
print(f"\nProcessed: {stats['processed']} unknown products")
print(f"Learned: {stats['learned']} products")
print(f"Success rate: {(stats['learned']/stats['processed']*100):.1f}%")

print(f"\n--- Filtering ---")
print(f"Too large (>{MAX_BOX_AREA*100:.0f}%): {stats['skipped_too_large']}")
print(f"Too small (<{MIN_BOX_AREA*100:.0f}%): {stats['skipped_too_small']}")
print(f"Low confidence: {stats['skipped_low_conf']}")
print(f"No valid boxes: {stats['skipped_no_boxes']}")

print(f"\n--- Clustering ---")
print(f"No cluster found: {stats['skipped_no_cluster']}")
print(f"Contaminated (appears in wrong images): {stats['skipped_contaminated']}")

print(f"\n--- Exclusivity Quality ---")
print(f"Perfect (100%): {stats['perfect_exclusivity']}")
print(f"High (>90%): {stats['high_exclusivity']}")
print(f"Medium (80-90%): {stats['medium_exclusivity']}")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print("="*60)
print("\n[+] Done! Simple exclusive clustering complete.")
print("[*] Key insight: Clusters ONLY accepted if they appear exclusively")
print("[*] in images containing this product - no contamination allowed!")
print("="*60)