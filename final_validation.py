#!/usr/bin/env python3
"""
Final Validation Script
Audits auto-labels against metadata ground truth to verify accuracy.
"""

import json
from pathlib import Path


def validate_labels():
    """Compare auto-labeled products against metadata ground truth."""
    
    mapping_path = Path("data/product_mapping.json")
    metadata_dir = Path("data/working/metadata")
    
    if not mapping_path.exists():
        print("No product_mapping.json found!")
        return
    
    with open(mapping_path) as f:
        mapping = json.load(f)
    
    stats = {
        'total': 0, 
        'correct': 0, 
        'incorrect': 0, 
        'invalid_format': 0,
        'no_metadata': 0, 
        'by_source': {}
    }
    incorrect_examples = []
    
    for img_name, products in mapping.items():
        # Get metadata for this image
        meta_path = metadata_dir / f"{Path(img_name).stem}.json"
        if not meta_path.exists():
            stats['no_metadata'] += 1
            continue
        
        with open(meta_path) as f:
            meta = json.load(f)
        meta_products = set(meta.get('BIN_FCSKU_DATA', {}).keys())
        
        for prod in products:
            source = prod.get('source', 'unknown')
            if source not in stats['by_source']:
                stats['by_source'][source] = {'correct': 0, 'incorrect': 0}
            stats['total'] += 1
            
            labeled_id = prod.get('original_product_id', '')
            
            # Skip old numeric labels (not ASINs)
            if not labeled_id.startswith('B0'):
                stats['invalid_format'] += 1
                continue
            
            if labeled_id in meta_products:
                stats['correct'] += 1
                stats['by_source'][source]['correct'] += 1
            else:
                stats['incorrect'] += 1
                stats['by_source'][source]['incorrect'] += 1
                if len(incorrect_examples) < 10:
                    incorrect_examples.append({
                        'image': img_name,
                        'labeled': labeled_id,
                        'metadata': list(meta_products),
                        'source': source
                    })
    
    # Print results
    valid_total = stats['correct'] + stats['incorrect']
    
    print("="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Total labels: {stats['total']}")
    print(f"  Invalid format (old numeric): {stats['invalid_format']}")
    print(f"  Valid ASIN labels: {valid_total}")
    print(f"")
    print(f"ASIN Label Accuracy:")
    print(f"  Correct (in metadata): {stats['correct']} ({stats['correct']/max(1,valid_total)*100:.1f}%)")
    print(f"  Incorrect: {stats['incorrect']} ({stats['incorrect']/max(1,valid_total)*100:.1f}%)")
    print(f"  Images without metadata: {stats['no_metadata']}")
    
    print(f"\nAccuracy by source:")
    for source, counts in sorted(stats['by_source'].items()):
        total = counts['correct'] + counts['incorrect']
        acc = counts['correct'] / max(1, total) * 100
        print(f"  {source}: {counts['correct']}/{total} correct ({acc:.1f}%)")
    
    if incorrect_examples:
        print(f"\nIncorrect examples (first 10):")
        for ex in incorrect_examples:
            print(f"  {ex['image']}:")
            print(f"    Labeled: '{ex['labeled']}'")
            print(f"    Metadata: {ex['metadata']}")
            print(f"    Source: {ex['source']}")
    
    # Summary
    print(f"\n{'='*60}")
    if stats['incorrect'] == 0:
        print("All labels match metadata!")
    else:
        print(f"WARNING: {stats['incorrect']} labels don't match metadata")
    
    return stats


if __name__ == "__main__":
    validate_labels()
