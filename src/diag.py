"""Diagnostic script to analyze clustering data and understand filtering"""

import json
from pathlib import Path
from collections import defaultdict

# Paths
CLUSTER_FILE = "clustered_output/product_mapping.json"
METADATA_DIR = Path("data/working/metadata")
IMAGES_DIR = Path("data/snapshots/binsense/images")

def analyze_clustering_data():
    """Analyze the clustering output to understand what we have"""
    
    print("\n" + "="*70)
    print("CLUSTERING DATA ANALYSIS")
    print("="*70)
    
    # Load clustering data
    with open(CLUSTER_FILE, 'r') as f:
        cluster_data = json.load(f)
    
    print(f"\nTotal images in clustering output: {len(cluster_data)}")
    
    # Statistics
    total_crops = 0
    conf_distribution = defaultdict(int)
    single_product_images = 0
    multi_product_images = 0
    images_with_metadata = 0
    images_without_metadata = 0
    single_product_with_high_conf = 0
    
    product_counts = defaultdict(int)
    crops_per_image = []
    
    for image_id, products in cluster_data.items():
        total_crops += len(products)
        crops_per_image.append(len(products))
        
        # Check metadata
        meta_file = METADATA_DIR / f"{image_id}.json"
        has_metadata = meta_file.exists()
        
        if has_metadata:
            images_with_metadata += 1
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            ground_truth_ids = set(meta.get('BIN_FCSKU_DATA', {}).keys())
            
            # Single vs multi product
            if len(ground_truth_ids) == 1:
                single_product_images += 1
                # Check if any crops have high confidence
                high_conf_crops = [p for p in products if p.get('yolo_conf', 0) >= 0.3]
                if high_conf_crops:
                    single_product_with_high_conf += 1
                    for p in high_conf_crops:
                        product_id = list(ground_truth_ids)[0]
                        product_counts[product_id] += 1
            else:
                multi_product_images += 1
                # For multi-product, only count matching predictions
                for p in products:
                    if p.get('yolo_conf', 0) >= 0.3 and p['original_product_id'] in ground_truth_ids:
                        product_counts[p['original_product_id']] += 1
        else:
            images_without_metadata += 1
        
        # Confidence distribution
        for p in products:
            conf = p.get('yolo_conf', 0)
            if conf >= 0.8:
                conf_distribution['0.8-1.0'] += 1
            elif conf >= 0.6:
                conf_distribution['0.6-0.8'] += 1
            elif conf >= 0.4:
                conf_distribution['0.4-0.6'] += 1
            elif conf >= 0.3:
                conf_distribution['0.3-0.4'] += 1
            elif conf >= 0.2:
                conf_distribution['0.2-0.3'] += 1
            else:
                conf_distribution['<0.2'] += 1
    
    print(f"\n{'='*70}")
    print("CROP STATISTICS:")
    print(f"  Total crops: {total_crops}")
    print(f"  Avg crops per image: {sum(crops_per_image)/len(crops_per_image):.1f}")
    print(f"  Max crops in one image: {max(crops_per_image)}")
    print(f"  Min crops in one image: {min(crops_per_image)}")
    
    print(f"\n{'='*70}")
    print("CONFIDENCE DISTRIBUTION:")
    for conf_range in ['0.8-1.0', '0.6-0.8', '0.4-0.6', '0.3-0.4', '0.2-0.3', '<0.2']:
        count = conf_distribution[conf_range]
        pct = count / total_crops * 100
        print(f"  {conf_range}: {count:4d} ({pct:5.1f}%)")
    
    high_conf_crops = sum(conf_distribution[r] for r in ['0.8-1.0', '0.6-0.8', '0.4-0.6', '0.3-0.4'])
    print(f"  >= 0.3: {high_conf_crops} ({high_conf_crops/total_crops*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("IMAGE METADATA:")
    print(f"  Images with metadata: {images_with_metadata}")
    print(f"  Images without metadata: {images_without_metadata}")
    print(f"  Single-product images: {single_product_images}")
    print(f"  Multi-product images: {multi_product_images}")
    print(f"  Single-product with high-conf crops: {single_product_with_high_conf}")
    
    print(f"\n{'='*70}")
    print("POTENTIAL CROPS AFTER FILTERING:")
    total_potential = sum(product_counts.values())
    print(f"  Total crops (if we use current logic): {total_potential}")
    print(f"  Unique products: {len(product_counts)}")
    print(f"  Avg crops per product: {total_potential/max(1, len(product_counts)):.1f}")
    
    # Show top products by crop count
    print(f"\n{'='*70}")
    print("TOP 10 PRODUCTS BY CROP COUNT:")
    sorted_products = sorted(product_counts.items(), key=lambda x: x[1], reverse=True)
    for i, (product_id, count) in enumerate(sorted_products[:10], 1):
        print(f"  {i:2d}. {product_id}: {count} crops")
    
    # Show products with very few crops
    low_crop_products = [(pid, cnt) for pid, cnt in product_counts.items() if cnt <= 2]
    print(f"\n{'='*70}")
    print(f"PRODUCTS WITH ≤2 CROPS: {len(low_crop_products)}")
    if low_crop_products:
        print(f"  Examples:")
        for product_id, count in low_crop_products[:10]:
            print(f"    {product_id}: {count} crop(s)")
    
    print(f"\n{'='*70}")
    print("RECOMMENDATION:")
    print(f"  Expected crops: {total_potential}")
    print(f"  Your result: 549")
    if total_potential > 549:
        print(f"  ⚠️  MISSING {total_potential - 549} crops!")
        print(f"  Check:")
        print(f"    1. Are all images present in {IMAGES_DIR}?")
        print(f"    2. Are manual annotations being included?")
        print(f"    3. Is CLIP model loading correctly?")
    
    # Check if images exist
    print(f"\n{'='*70}")
    print("CHECKING IMAGE FILES:")
    missing_images = 0
    for image_id in list(cluster_data.keys())[:100]:  # Check first 100
        image_file = IMAGES_DIR / f"{image_id}.jpg"
        if not image_file.exists():
            missing_images += 1
    
    if missing_images > 0:
        print(f"  ⚠️  WARNING: {missing_images}/100 sample images are missing!")
    else:
        print(f"  ✓ All sampled images exist")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    analyze_clustering_data()