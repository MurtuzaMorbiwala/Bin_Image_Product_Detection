#!/usr/bin/env python3
"""
Deploy BinSense to Kaggle with automated deps build

TYPICAL USAGE:
  One-time setup:
    python deploy.py --setup
    (then manually run deps kernel on Kaggle)
    python deploy.py --upload-dataset --use-existing-deps
    
  Regular updates (just kernel code):
    python deploy.py --kernel-only
"""

import sys
import json
import os
os.environ["PYTHONUTF8"] = "1"

import time
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from kaggle.api.kaggle_api_extended import KaggleApi


class LoopDeployer:
    def __init__(self):
        self.api = KaggleApi()
        self.api.authenticate()
        config = Path.home() / ".kaggle" / "kaggle.json"
        self.username = json.load(open(config))["username"]

        self.dataset_slug = "binsense-loop"
        self.kernel_slug = "binsense-training-loop"
        self.deps_kernel_slug = "binsense-build-deps"

    # ---------------- DEPS KERNEL (ONE-TIME) ----------------

    def push_deps_kernel(self):
        """Push a kernel that builds deps on Kaggle (ONE-TIME)"""
        print("\n" + "="*60)
        print("PUSHING DEPS BUILD KERNEL (ONE-TIME SETUP)")
        print("="*60)
        
        kernel_dir = Path(tempfile.mkdtemp())

        try:
            # Build script that downloads deps AND uploads them as a dataset
            script = f"""
import subprocess
import json
from pathlib import Path

# Download dependencies
deps_dir = Path("/kaggle/working/deps")
deps_dir.mkdir(exist_ok=True)
pkgs = [
    "ultralytics",
    "ftfy",
    "regex",
    "pillow",
    "numpy==1.26.4",
    "git+https://github.com/openai/CLIP.git"
]
subprocess.check_call(["pip", "download", *pkgs, "-d", str(deps_dir)])
print("Deps downloaded to /kaggle/working/deps")

# Create dataset metadata
meta = {{
    "title": "BinSense Dependencies",
    "id": "{self.username}/binsense-deps",
    "licenses": [{{"name": "CC0-1.0"}}]
}}
with open("/kaggle/working/dataset-metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print("Created dataset metadata for upload")
print("Dataset will be available at: https://www.kaggle.com/datasets/{self.username}/binsense-deps")
"""
            (kernel_dir / "build_deps.py").write_text(script)

            meta = {
                "id": f"{self.username}/{self.deps_kernel_slug}",
                "title": "Build BinSense Deps",
                "code_file": "build_deps.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_internet": True,
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": []
            }

            (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
            self.api.kernels_push(str(kernel_dir))
            
            print("✓ Deps build kernel pushed successfully")
            print("\n" + "="*60)
            print("NEXT STEPS:")
            print("="*60)
            print(f"1. Visit: https://www.kaggle.com/code/{self.username}/{self.deps_kernel_slug}")
            print("2. Click 'Run' button")
            print("3. Wait for completion (~5-10 minutes)")
            print("4. The kernel will output the deps to /kaggle/working/")
            print("5. Manually create a dataset from this output:")
            print(f"   - Go to the Output tab")
            print(f"   - Click 'New Dataset' and create 'binsense-deps'")
            print(f"6. Then run: python {sys.argv[0]} --upload-dataset")
            print("="*60)
            
        finally:
            shutil.rmtree(kernel_dir, ignore_errors=True)

    def download_deps(self):
        """Download deps dataset from Kaggle"""
        print("\nDownloading deps dataset from Kaggle...")
        deps_dataset = f"{self.username}/binsense-deps"
        output_dir = Path("data/deps_output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Download the deps dataset
            self.api.dataset_download_files(deps_dataset, path=str(output_dir), unzip=True)
            print(f"✓ Deps dataset downloaded to {output_dir}")
            
            # Check for deps folder
            deps_path = output_dir / "deps"
            if deps_path.exists():
                return deps_path
            
            # Maybe the deps are directly in the output folder
            # Check if .whl files exist directly in output_dir
            whl_files = list(output_dir.glob("*.whl"))
            if whl_files:
                print(f"✓ Found {len(whl_files)} wheel files in output directory")
                return output_dir
            
            print(f"\n✗ ERROR: No deps found in {output_dir}")
            print(f"Contents of {output_dir}:")
            for item in output_dir.iterdir():
                print(f"  - {item}")
            sys.exit(1)
            
        except Exception as e:
            print(f"\n✗ ERROR downloading deps dataset: {e}")
            print("\n" + "="*60)
            print("DEPS DATASET NOT FOUND")
            print("="*60)
            print("You need to create the deps dataset first:")
            print(f"1. Visit: https://www.kaggle.com/code/{self.username}/{self.deps_kernel_slug}")
            print("2. Make sure it has run successfully")
            print("3. Go to the Output tab")
            print("4. Click 'New Dataset' button")
            print(f"5. Create a dataset named 'binsense-deps'")
            print(f"6. Then run: python {sys.argv[0]} --upload-dataset")
            print("="*60)
            sys.exit(1)

    # ---------------- DATASET (ONE-TIME) ----------------

    def prepare_dataset(self, use_existing_deps=False):
        """Prepare dataset snapshot with deps (ONE-TIME)"""
        print("\n" + "="*60)
        print("PREPARING DATASET (ONE-TIME SETUP)")
        print("="*60)
        
        work = Path("data/working")
        snap = Path("data/snapshots/binsense")
        shutil.rmtree(snap, ignore_errors=True)
        snap.mkdir(parents=True)

        # Validate working directory
        if not work.exists():
            print(f"✗ ERROR: Working directory {work} does not exist")
            sys.exit(1)
        
        required_dirs = ["images", "annotations", "metadata"]
        for dir_name in required_dirs:
            if not (work / dir_name).exists():
                print(f"✗ ERROR: Required directory {work}/{dir_name} does not exist")
                sys.exit(1)

        # Copy data
        print("Copying data files...")
        shutil.copytree(work / "images", snap / "images")
        shutil.copytree(work / "annotations", snap / "labels")
        shutil.copytree(work / "metadata", snap / "metadata")

        # Copy metadata & weights
        for f in ["product_mapping.json", "best.pt"]:
            src = Path("data") / f
            if src.exists():
                shutil.copy(src, snap / f)
                print(f"  ✓ Copied {f}")

        # Handle deps
        if use_existing_deps:
            # Use manually downloaded deps
            deps_source = Path("data/deps_output")
            
            # Check if it's in deps subfolder or directly in output
            if (deps_source / "deps").exists():
                deps_to_copy = deps_source / "deps"
            elif list(deps_source.glob("*.whl")):
                deps_to_copy = deps_source
            else:
                print(f"\n✗ ERROR: Deps not found at {deps_source.absolute()}")
                print("Please download deps dataset first")
                sys.exit(1)
                
            print(f"\n✓ Using existing deps from {deps_to_copy}")
            shutil.copytree(deps_to_copy, snap / "deps")
        else:
            # Download deps dataset from Kaggle
            print("\nDownloading deps dataset from Kaggle...")
            deps = self.download_deps()
            shutil.copytree(deps, snap / "deps")
            print("  ✓ Deps added to dataset")

        # Create data.yaml
        with open(snap / "data.yaml", "w") as f:
            f.write(
                "path: /kaggle/working/data\n"
                "train: images\n"
                "val: images\n"
                "nc: 1\n"
                "names: ['product']\n"
            )

        # Create dataset metadata
        meta = {
            "title": "BinSense Loop (Data + Deps)",
            "id": f"{self.username}/{self.dataset_slug}",
            "licenses": [{"name": "CC0-1.0"}]
        }
        with open(snap / "dataset-metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"✓ Dataset prepared at {snap}")
        return snap

    def upload_dataset(self, snap: Path):
        """Upload dataset to Kaggle"""
        print("\nUploading dataset to Kaggle...")
        try:
            self.api.dataset_status(f"{self.username}/{self.dataset_slug}")
            self.api.dataset_create_version(
                str(snap),
                version_notes=f"Updated {datetime.now():%Y-%m-%d %H:%M}",
                dir_mode="zip"
            )
            print("✓ Dataset updated")
        except:
            self.api.dataset_create_new(str(snap), dir_mode="zip")
            print("✓ Dataset created")
        
        print(f"\nDataset URL: https://www.kaggle.com/datasets/{self.username}/{self.dataset_slug}")

    # ---------------- KERNEL (REGULAR UPDATES) ----------------

    def push_kernel(self, debug=False):
        """Push training kernel - USE THIS FOR REGULAR UPDATES"""
        print("\n" + "="*60)
        print("PUSHING TRAINING KERNEL")
        print("="*60)
        
        kernel_dir = Path(tempfile.mkdtemp())
        try:
            meta = {
                "id": f"{self.username}/{self.kernel_slug}",
                "title": "BinSense Training Loop",
                "code_file": "kaggle_loop.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": False,
                "dataset_sources": [f"{self.username}/{self.dataset_slug}"],
                "competition_sources": [],
                "kernel_sources": []
            }

            (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
            
            # Look for kaggle_loop.py in current dir or training_loop/ subdirectory
            loop_file = None
            if Path("kaggle_loop.py").exists():
                loop_file = Path("kaggle_loop.py")
            elif Path("training_loop/kaggle_loop.py").exists():
                loop_file = Path("training_loop/kaggle_loop.py")
            
            if not loop_file:
                print("✗ ERROR: kaggle_loop.py not found")
                print("  Looked in:")
                print("    - ./kaggle_loop.py")
                print("    - ./training_loop/kaggle_loop.py")
                sys.exit(1)
                
            shutil.copy(loop_file, kernel_dir / "kaggle_loop.py")

            print(f"Uploading kernel (debug={debug})...")
            self.api.kernels_push(str(kernel_dir))
            print("✓ Training kernel pushed successfully")
            print(f"\nKernel URL: https://www.kaggle.com/code/{self.username}/{self.kernel_slug}")
            
        finally:
            shutil.rmtree(kernel_dir, ignore_errors=True)


# ---------------- MAIN ----------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Deploy BinSense to Kaggle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
WORKFLOW:

ONE-TIME SETUP (run these once):
  1. python deploy.py --setup
     → Pushes deps build kernel to Kaggle
     
  2. Manually run the deps kernel on Kaggle website
     → Go to the URL shown and click "Run"
     
  3. python deploy.py --upload-dataset --use-existing-deps
     → Downloads deps and uploads complete dataset

REGULAR UPDATES (run this whenever you change kaggle_loop.py):
  python deploy.py --kernel-only
  → Updates just the training kernel code
        """
    )
    
    # One-time setup
    parser.add_argument("--setup", action="store_true",
                        help="ONE-TIME: Push deps build kernel (then run it manually on Kaggle)")
    parser.add_argument("--upload-dataset", action="store_true",
                        help="ONE-TIME: Download deps and upload dataset")
    
    # Regular updates
    parser.add_argument("--kernel-only", action="store_true",
                        help="REGULAR: Update training kernel code only")
    
    # Advanced
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode for training kernel")
    parser.add_argument("--use-existing-deps", action="store_true",
                        help="Use manually downloaded deps from data/deps_output/deps")
    
    args = parser.parse_args()

    deployer = LoopDeployer()

    if args.setup:
        # ONE-TIME: Push deps kernel
        deployer.push_deps_kernel()
        
    elif args.upload_dataset:
        # ONE-TIME: Prepare and upload dataset with deps
        snap = deployer.prepare_dataset(use_existing_deps=args.use_existing_deps)
        deployer.upload_dataset(snap)
        print("\n" + "="*60)
        print("DATASET SETUP COMPLETE!")
        print("="*60)
        print("Now you can update your training kernel anytime with:")
        print(f"  python {sys.argv[0]} --kernel-only")
        print("="*60)
        
    elif args.kernel_only:
        # REGULAR: Just update the kernel
        deployer.push_kernel(debug=args.debug)
        
    else:
        parser.print_help()
        print("\n" + "="*60)
        print("QUICK START:")
        print("="*60)
        print("First time? Run these in order:")
        print(f"  1. python {sys.argv[0]} --setup")
        print("  2. Run deps kernel on Kaggle (link will be shown)")
        print(f"  3. python {sys.argv[0]} --upload-dataset --use-existing-deps")
        print("\nAfter setup, just use:")
        print(f"  python {sys.argv[0]} --kernel-only")
        print("="*60)


if __name__ == "__main__":
    main()