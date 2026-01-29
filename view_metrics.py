#!/usr/bin/env python3
"""
View training metrics history
"""
import json
from pathlib import Path
from datetime import datetime

def view_metrics():
    """Display metrics history in a formatted table"""
    history_file = Path("data/metrics_history.json")
    
    if not history_file.exists():
        print("No metrics history found. Run training first!")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    if not history:
        print("Metrics history is empty.")
        return
    
    print("\n" + "="*100)
    print("TRAINING METRICS HISTORY")
    print("="*100)
    
    # Header
    print(f"\n{'#':<4} {'Date':<20} {'Kernel':<30} {'mAP@50':<10} {'mAP@50-95':<12} {'Precision':<12} {'Recall':<10}")
    print("-"*100)
    
    # Rows
    for i, entry in enumerate(history, 1):
        timestamp = datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        kernel = entry.get('kernel_slug', 'unknown')[:28]
        metrics = entry['metrics']
        
        print(f"{i:<4} {timestamp:<20} {kernel:<30} "
              f"{metrics['map50']:<10.4f} {metrics['map50_95']:<12.4f} "
              f"{metrics['precision']:<12.4f} {metrics['recall']:<10.4f}")
    
    # Latest metrics detail
    if history:
        latest = history[-1]
        print("\n" + "="*100)
        print("LATEST TRAINING DETAILS")
        print("="*100)
        print(f"Timestamp:    {latest['timestamp']}")
        print(f"Kernel:       {latest['kernel_slug']}")
        print(f"Dataset:      {latest['dataset']}")
        print(f"Images:       {latest['images']}")
        print(f"Labels:       {latest['labels']}")
        print(f"Total Boxes:  {latest['total_boxes']}")
        print(f"Epochs:       {latest['epochs']}")
        print(f"Batch Size:   {latest['batch_size']}")
        print(f"Model:        {latest['model']}")
        print(f"\nMetrics:")
        print(f"  mAP@50:     {latest['metrics']['map50']:.4f}")
        print(f"  mAP@50-95:  {latest['metrics']['map50_95']:.4f}")
        print(f"  Precision:  {latest['metrics']['precision']:.4f}")
        print(f"  Recall:     {latest['metrics']['recall']:.4f}")
        
        # Show improvement if there's history
        if len(history) > 1:
            prev = history[-2]
            print(f"\nChange from previous:")
            print(f"  mAP@50:     {latest['metrics']['map50'] - prev['metrics']['map50']:+.4f}")
            print(f"  mAP@50-95:  {latest['metrics']['map50_95'] - prev['metrics']['map50_95']:+.4f}")
            print(f"  Precision:  {latest['metrics']['precision'] - prev['metrics']['precision']:+.4f}")
            print(f"  Recall:     {latest['metrics']['recall'] - prev['metrics']['recall']:+.4f}")
    
    print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    view_metrics()
