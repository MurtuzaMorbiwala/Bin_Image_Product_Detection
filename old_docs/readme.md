# BinSense2: Intelligent Product Detection System

## Overview

BinSense2 is an automated product detection and identification system designed for warehouse bin images. The system uses a **two-stage approach** combining object detection with product recognition to identify 5,285+ unique products across 3,874 images.

### Key Features
- **Two-Stage Detection**: Generic object detection + product identification
- **Browser-Based Annotation**: Flask web app with click-and-drag bounding boxes
- **Iterative Training**: Self-improving model with active learning
- **Cloud Training**: Kaggle GPU-powered training pipeline
- **Rich Metadata**: Product information from Amazon bin images

### Dataset Statistics
- **Images**: 3,874 bin images
- **Unique Products**: 5,285 products
- **Total Instances**: 10,137 product occurrences
- **Products per Image**: 2.62 average (range: 0-5)
- **Multi-Image Products**: 2,837 products (53.7%) appear in multiple images

---

## Project Structure

```
BinSense2/
├── data/
│   ├── working/              # Active annotation workspace
│   │   ├── images/           # 3,874 bin images (.jpg)
│   │   ├── metadata/         # JSON files with product ASINs and names
│   │   └── annotations/      # YOLO format annotations (.txt)
│   └── snapshots/            # Frozen dataset versions for training
│
├── src/
│   ├── annotation_app_flask.py    # Browser-based annotation tool (Flask + HTML5 Canvas)
│   ├── annotation_app_canvas.py   # Legacy Streamlit app (deprecated)
│   └── train.py                   # Kaggle-compatible YOLO training script
│
├── generate_labels.py         # Generate class ID mappings from metadata
├── label_mapping.json         # Product name → class ID mapping (5,285 classes)
├── products_by_image.json     # Image ID → Products with class IDs
├── check_products.py          # Product statistics and analysis
├── requirements-local.txt     # Local annotation dependencies (Flask, OpenCV)
├── requirements-kaggle.txt    # Training dependencies (Ultralytics, PyTorch)
└── kaggle_deploy.py           # Automated Kaggle deployment script
```

---

## Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repository-url>
cd BinSense2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install local dependencies
pip install -r requirements-local.txt
```

### 2. Generate Label Mappings
```bash
# Create class ID mappings from metadata
python generate_labels.py

# This generates:
# - label_mapping.json (product name → class ID)
# - products_by_image.json (image → products with class IDs)
```

### 3. Start Annotation Tool
```bash
# Launch Flask annotation app
python src/annotation_app_flask.py

# Open browser to: http://localhost:5000
```

### 4. Train on Kaggle (Optional)
```bash
# Deploy training script to Kaggle
python kaggle_deploy.py

# In Kaggle notebook:
!python train.py --data-path /kaggle/input/your-dataset --epochs 50
```

## Annotation Tool

### Flask Web Application

The annotation tool is a **browser-based Flask application** with HTML5 Canvas for intuitive mouse-based drawing.

#### Key Features
- **Click-and-Drag Drawing**: Draw bounding boxes directly on images
- **Keyboard Shortcuts**: 
  - `S` - Save bounding box
  - `D` - Delete bounding box
  - `N` - Next image
  - `P` - Previous image
- **Visual Feedback**:
  - **Green boxes for selected product (editable)
  - **Grey boxes for other products (with class ID and name)
- **Smart Editing**: Load existing annotations for modification
- **Auto-Save**: Saves to YOLO format (.txt files)
- **Real-time Updates**: See all annotations as you work

#### How to Use

1. **Start the app**:
   ```bash
   python src/annotation_app_flask.py
   ```

2. **Open browser**: Navigate to `http://localhost:5000`

3. **Annotation workflow**:
   - Select an image from the dropdown
   - Click a product from the list (shows existing box if annotated)
   - Click and drag on the image to draw/redraw bounding box
   - Press `S` to save
   - Press `N` to move to next image

4. **Output**: Annotations saved to `data/working/annotations/{image_id}.txt`

#### YOLO Format
```
class_id x_center y_center width height
```
Example:
```
1133 0.829167 0.501070 0.337500 0.991892
2321 0.329167 0.501070 0.650000 1.002703
```

---

Stage 1: Localization (The "Where")
Model: YOLO (v8/v11) trained as a class-agnostic detector.

Role: Identifies all generic "objects" within the bin. By focusing only on the product class, the model achieves high precision in complex, occluded environments.

Stage 2: Identification (The "What")
Model: CLIP-ViT (Vision Transformer).

Role: Extracts a 512-dimensional feature vector (fingerprint) from each crop provided by Stage 1.

Advantage: Unlike standard CNNs, the Transformer backbone is highly robust to the 360° rotations and varying angles of products in the bins.

🔄 The Self-Learning Lifecycle
Since the dataset lacks manual bounding box labels for all 500,000 images, we implement a Bootstrap Cycle:

Seed Phase: Manually label 100 "Gold Standard" images to train the initial Stage 1 detector.

Gallery Creation: Automatically extract reference features from "Single-Item Bins" (bins where metadata quantity = 1) to build the initial ASIN identity database.

Pseudo-Labeling: Run the pipeline on the full dataset. When the Stage 1 count matches the Metadata count and Stage 2 has high confidence (>0.90), the system automatically saves the result as a new training label.

Iterative Refinement: The models are periodically re-trained on this growing pool of verified data, allowing the AI to "teach itself" to recognize more products over time.

🛠️ Tech Stack
Development: VS Code

Compute: Kaggle GPU (Training & Batch Inference)

Models: Ultralytics (YOLO), OpenAI (CLIP-ViT)

Data Handling: Python (Pandas/SQL), OpenCV (Image Processing)