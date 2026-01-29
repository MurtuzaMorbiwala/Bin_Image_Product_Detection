#!/usr/bin/env python3
"""
Kaggle Deployment using Native Python API
Uses kaggle.api directly instead of CLI commands
"""

import os
os.environ["PYTHONUTF8"] = "1"

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Import Kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print(" Kaggle API not installed!")
    print("   Run: pip install kaggle")
    sys.exit(1)


class KaggleTrainer:
    """Deploy and train using Kaggle Python API"""
    
    def __init__(self):
        """Initialize Kaggle API"""
        print(" Initializing Kaggle API...")
        self.api = KaggleApi()
        self.api.authenticate()
        
        # Get username from credentials
        config_dir = Path.home() / '.kaggle'
        kaggle_json = config_dir / 'kaggle.json'
        
        with open(kaggle_json, 'r') as f:
            creds = json.load(f)
            self.username = creds['username']
        
        print(f" Authenticated as: {self.username}")
        
        self.dataset_slug = None
        self.kernel_slug = None
    
    def create_snapshot(self):
        """Create training snapshot"""
        print(f"\n{'='*60}")
        print(f"STEP 1: Creating Snapshot")
        print(f"{'='*60}\n")
        
        import subprocess
        result = subprocess.run(
            "python create_snapshot.py",
            shell=True,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f" Snapshot creation failed:")
            print(result.stderr)
            raise RuntimeError("Snapshot creation failed")
        
        print(result.stdout)
        return Path("data/snapshots/binsense")
    
    def create_dataset(self, update=True):
        """Create or update dataset using Kaggle API"""
        print(f"\n{'='*60}")
        print(f"STEP 2: Updating Dataset on Kaggle")
        print(f"{'='*60}\n")
        
        snapshot_dir = Path("data/snapshots/binsense")
        self.dataset_slug = "binsensefixed"
        
        # Create dataset metadata
        metadata = {
            "title": "BinSense Product Detection",
            "id": f"{self.username}/{self.dataset_slug}",
            "licenses": [{"name": "CC0-1.0"}],
            "description": f"Annotated warehouse bin images for YOLO training. Created: {datetime.now().isoformat()}"
        }
        
        metadata_path = snapshot_dir / "dataset-metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f" Dataset: {self.username}/{self.dataset_slug}")
        
        print(" Updating existing dataset...")
        self.api.dataset_create_version(
            folder=str(snapshot_dir),
            version_notes="Updated annotations",
            dir_mode='zip',
            quiet=False
        )
        print(" Dataset updated")
        
        dataset_url = f"https://www.kaggle.com/datasets/{self.username}/{self.dataset_slug}"
        print(f" Dataset URL: {dataset_url}")
        
        return dataset_url
    
    def push_kernel(self, epochs=50, batch=16, model_size='n', force_new=False):
        """Push training script as Kaggle kernel using API"""
        print(f"\n{'='*60}")
        print(f"STEP 3: Pushing Training Script to Kaggle")
        print(f"{'='*60}\n")
        
        # Reuse existing kernel to avoid reinstalling dependencies
        # Only create new kernel if force_new=True or first time
        if force_new:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.kernel_slug = f"binsense-training-{timestamp}"
            kernel_title = f"BinSense Training {timestamp}"
            print(f"[*] Creating new kernel: {self.kernel_slug}")
        else:
            self.kernel_slug = "binsense-training-main"
            kernel_title = "BinSense Training Main"  # Title must match slug
            print(f"[*] Updating existing kernel: {self.kernel_slug}")
        
        # Create temporary kernel directory
        import tempfile
        import shutil
        kernel_dir = Path(tempfile.mkdtemp(prefix="kaggle_kernel_"))
        
        try:
            # Create kernel metadata
            dataset_source = f"{self.username}/{self.dataset_slug}"
            metadata = {
                "id": f"{self.username}/{self.kernel_slug}",
                "title": kernel_title,
                "code_file": "kaggle_train_script.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [dataset_source],
                "competition_sources": [],
                "kernel_sources": [],
                "docker_image_pinning_type": "original",
                "environment_variables": {
                    "EPOCHS": str(epochs),
                    "BATCH_SIZE": str(batch),
                    "MODEL_SIZE": model_size,
                    "IMG_SIZE": "640"
                }
            }
            
            print(f"[*] Attaching dataset: {dataset_source}")
            
            # Save metadata with UTF-8
            metadata_path = kernel_dir / "kernel-metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Copy training script with UTF-8 encoding
            script_path = Path("kaggle_train_script.py")
            if not script_path.exists():
                raise FileNotFoundError("kaggle_train_script.py not found!")
            
            # Read and write with explicit UTF-8 encoding
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            dest_script = kernel_dir / "kaggle_train_script.py"
            with open(dest_script, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            print(f" Kernel: {self.username}/{self.kernel_slug}")
            print(" Pushing kernel...")
            
            # Push kernel using API from temp directory
            self.api.kernels_push(str(kernel_dir))
            
            print(" Kernel pushed")
            
        finally:
            # Cleanup temp directory
            shutil.rmtree(kernel_dir, ignore_errors=True)
        
        kernel_url = f"https://www.kaggle.com/code/{self.username}/{self.kernel_slug}"
        print(f" Kernel URL: {kernel_url}")
        
        return kernel_url
    
    def check_kernel_status(self):
        """Check kernel execution status"""
        try:
            status = self.api.kernels_status(f"{self.username}/{self.kernel_slug}")
            return status
        except Exception as e:
            return {"status": "unknown", "error": str(e)}
    
    def download_kernel_output(self, output_dir="kaggle_output"):
        """Download kernel output using subprocess to avoid encoding issues"""
        print(f"\n{'='*60}")
        print(f"STEP 4: Downloading Training Output")
        print(f"{'='*60}\n")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"[*] Downloading output from {self.username}/{self.kernel_slug}...")
        
        # Use subprocess with kaggle CLI to avoid Python encoding issues
        import subprocess
        
        try:
            # Download kernel output using CLI
            result = subprocess.run(
                f'kaggle kernels output {self.username}/{self.kernel_slug} -p "{output_path}"',
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'  # Ignore encoding errors
            )
            
            if result.returncode == 0:
                print("[+] Download completed")
            else:
                print(f"[!] Download warning: {result.stderr if result.stderr else 'Check output directory'}")
                
        except Exception as e:
            print(f"[!] Download error (files may still have downloaded): {e}")
        
        # Download logs separately
        print(f"\n[*] Downloading kernel logs...")
        try:
            result = subprocess.run(
                f'kaggle kernels output {self.username}/{self.kernel_slug} -p "{output_path}" --log',
                shell=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='ignore'
            )
            if result.returncode == 0:
                print("[+] Logs downloaded")
        except Exception as e:
            print(f"[!] Log download error: {e}")
        
        # List downloaded files
        files = list(output_path.glob('*'))
        print(f"\n[+] Downloaded {len(files)} files:")
        for f in files:
            if f.is_file():
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"   - {f.name} ({size_mb:.2f} MB)")
        
        # Check for best.pt
        best_pt = output_path / "best.pt"
        if best_pt.exists():
            print(f"\n[+] Trained model: {best_pt}")
            print(f"   Ready to use for inference!")
        else:
            print(f"\n[!] best.pt not found. Check if training completed successfully.")
        
        # Parse and save metrics to history
        summary_file = output_path / "training_summary.json"
        if summary_file.exists():
            self._save_metrics_to_history(summary_file)
        
        return output_path
    
    def _save_metrics_to_history(self, summary_file):
        """Save training metrics to a local history file"""
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            # Create metrics history file
            history_file = Path("data/metrics_history.json")
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load existing history
            history = []
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
            
            # Add timestamp and kernel info
            from datetime import datetime
            entry = {
                'timestamp': datetime.now().isoformat(),
                'kernel_slug': self.kernel_slug,
                **summary
            }
            
            history.append(entry)
            
            # Save updated history
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            print(f"\n[+] Metrics saved to: {history_file}")
            print(f"\n📊 Training Metrics:")
            print(f"   mAP@50:     {summary['metrics']['map50']:.4f}")
            print(f"   mAP@50-95:  {summary['metrics']['map50_95']:.4f}")
            print(f"   Precision:  {summary['metrics']['precision']:.4f}")
            print(f"   Recall:     {summary['metrics']['recall']:.4f}")
            
        except Exception as e:
            print(f"[!] Could not save metrics to history: {e}")
    
    def full_workflow(self, epochs=400, batch=16, model_size='n', 
                     wait_for_completion=False, force_new=False):
        """Complete workflow using Kaggle API"""
        print(f"\n{'#'*60}")
        print(f"# KAGGLE API TRAINING WORKFLOW")
        print(f"{'#'*60}\n")
        
        try:
            # Step 1: Create snapshot
            self.create_snapshot()
            
            # Step 2: Update dataset
            dataset_url = self.create_dataset()
            
            # Wait for dataset to be ready
            print("\n Waiting for dataset to be ready...")
            time.sleep(10)
            
            # Step 3: Push training script
            kernel_url = self.push_kernel(epochs, batch, model_size, force_new=force_new)
            
            print(f"\n{'='*60}")
            print(f" DEPLOYMENT COMPLETED!")
            print(f"{'='*60}\n")
            print(f" Dataset: {dataset_url}")
            print(f" Kernel: {kernel_url}")
            print(f"\n Training has been queued on Kaggle!")
            
            if wait_for_completion:
                print(f"\n Monitoring training progress...")
                print(f"   (This may take 30-60 minutes)")
                
                while True:
                    status_info = self.check_kernel_status()
                    status = status_info.get('status', 'unknown')
                    
                    print(f"   Status: {status}")
                    
                    if status in ['complete', 'error', 'failed', 'cancelled']:
                        break
                    
                    time.sleep(60)  # Check every minute
                
                if status == 'complete':
                    print(f"\n Training completed successfully!")
                    
                    # Download output
                    print(f"\n Downloading results...")
                    time.sleep(5)  # Wait a bit for files to be ready
                    self.download_kernel_output()
                else:
                    print(f"\n Training ended with status: {status}")
                    print(f"   Check kernel logs at: {kernel_url}")
            else:
                print(f"\n To download results later, run:")
                print(f"   python kaggle_deploy_api.py --download-only")
            
            return {
                'dataset_url': dataset_url,
                'kernel_url': kernel_url,
                'dataset_slug': self.dataset_slug,
                'kernel_slug': self.kernel_slug
            }
            
        except Exception as e:
            print(f"\n Workflow failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy and train using Kaggle Python API')
    parser.add_argument('--epochs', type=int, default=400, help='Training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size')
    parser.add_argument('--wait', action='store_true', help='Wait for training to complete')
    parser.add_argument('--force-new', action='store_true', 
                       help='Force create new kernel instead of updating existing')
    
    # Individual actions
    parser.add_argument('--snapshot-only', action='store_true', help='Only create snapshot')
    parser.add_argument('--upload-only', action='store_true', help='Only upload dataset')
    parser.add_argument('--push-only', action='store_true', help='Only push kernel')
    parser.add_argument('--download-only', action='store_true', help='Only download output')
    
    args = parser.parse_args()
    
    try:
        trainer = KaggleTrainer()
        
        if args.snapshot_only:
            trainer.create_snapshot()
        elif args.upload_only:
            trainer.dataset_slug = "binsensefixed"
            trainer.create_dataset()
        elif args.push_only:
            trainer.dataset_slug = "binsensefixed"
            trainer.push_kernel(args.epochs, args.batch, args.model, force_new=args.force_new)
        elif args.download_only:
            # Find the latest binsense-training kernel
            print("[*] Finding latest training kernel...")
            kernels = trainer.api.kernels_list(user=trainer.username, search="binsense-training")
            if kernels:
                # Get the most recent one
                latest = kernels[0]
                trainer.kernel_slug = latest.ref.split('/')[-1]
                print(f"[+] Found: {trainer.kernel_slug}")
            else:
                trainer.kernel_slug = "binsense-training-main"
                print("[!] No kernels found, using default: binsense-training-main")
            trainer.download_kernel_output()
        else:
            # Full workflow
            trainer.full_workflow(
                epochs=args.epochs,
                batch=args.batch,
                model_size=args.model,
                wait_for_completion=args.wait,
                force_new=args.force_new
            )
        
        print("\n Success!")
        return 0
        
    except Exception as e:
        print(f"\n Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
