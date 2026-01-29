"""Run inference on images using the trained model"""
from ultralytics import YOLO
from pathlib import Path
import argparse
import sys

def run_inference(image_path, model_path="kaggle_output/best.pt", save=True):
    """Run inference on a single image or directory"""
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Run inference
    print(f"Running inference on: {image_path}")
    results = model(image_path, save=save, conf=0.25)
    
    # Print results
    for r in results:
        boxes = r.boxes
        print(f"\nDetections: {len(boxes)}")
        for i, box in enumerate(boxes):
            conf = box.conf[0].item()
            xyxy = box.xyxy[0].tolist()
            print(f"  Box {i+1}: confidence={conf:.2%}, bbox={[int(x) for x in xyxy]}")
    
    if save:
        print(f"\nResults saved to: runs/detect/predict*/")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run YOLO inference')
    parser.add_argument('image', nargs='?', help='Image path or directory')
    parser.add_argument('--model', default='kaggle_output/best.pt', help='Model path')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()
    
    if not args.image:
        # Demo mode - use a training image
        demo_images = list(Path("data/snapshots/binsense/images").glob("*.jpg"))[:3]
        if demo_images:
            print("Demo mode - running on sample training images\n")
            for img in demo_images:
                run_inference(str(img), args.model)
                print("-" * 50)
        else:
            print("Usage: python run_inference.py <image_path>")
            print("       python run_inference.py data/snapshots/binsense/images/00340.jpg")
    else:
        run_inference(args.image, args.model)

if __name__ == "__main__":
    main()
