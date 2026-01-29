import torch
from pathlib import Path

def check_best_pt(weights_path=r"C:\Users\bindi\PythonDev\BinSense2\data\best.pt"):  # Added 'r' prefix
    try:
        print(f"Loading weights from: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu")

        print("\n" + "="*60)
        print("CHECKPOINT TYPE:", type(ckpt))
        print("="*60)

        if isinstance(ckpt, dict):
            print("\nTop-level keys:")
            for k in ckpt.keys():
                print(f"  - {k}")

            print("\n--- VERSION INFO ---")
            for k in ["ultralytics_version", "yolo_version", "version"]:
                if k in ckpt:
                    print(f"{k}: {ckpt[k]}")

            print("\n--- MODEL INFO ---")
            model = ckpt.get("model", None)
            if model is not None:
                print("Model class:", model.__class__)
                print("Model type:", type(model))
            
            print("\n--- TRAIN ARGS ---")
            train_args = ckpt.get("train_args", None)
            if train_args:
                for k in ["model", "imgsz", "batch", "optimizer", "epochs"]:
                    if k in train_args:
                        print(f"{k}: {train_args[k]}")

            print("\n--- EPOCH ---")
            print("epoch:", ckpt.get("epoch"))

        else:
            print("⚠️ This checkpoint is NOT a dict (unexpected)")

    except Exception as e:
        print(f"\n❌ Error loading checkpoint: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Check YOLO model checkpoint')
    parser.add_argument('--weights', type=str, 
                      default=r"C:\Users\bindi\PythonDev\BinSense2\data\best.pt",
                      help='Path to the best.pt file')
    args = parser.parse_args()
    
    check_best_pt(args.weights)