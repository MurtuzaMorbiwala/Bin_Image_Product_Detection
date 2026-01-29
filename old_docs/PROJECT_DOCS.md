# BinSense2: Comprehensive Project Documentation

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Project Structure](#-project-structure)
- [Core Components](#-core-components)
- [File Descriptions](#-file-descriptions)
- [Getting Started](#-getting-started)
- [Training Pipeline](#-training-pipeline)
- [Inference and Deployment](#-inference-and-deployment)
- [Visualization Tools](#-visualization-tools)
- [Technical Stack](#-technical-stack)

## 🌟 Project Overview

BinSense2 is an advanced computer vision system designed for automated product detection and identification in warehouse bin images. The system employs a two-stage approach combining object detection with product recognition to identify thousands of unique products with high accuracy.

## ✨ Key Features

- **Two-Stage Detection**: Combines generic object detection with product identification
- **Active Learning**: Self-improving model with automated data labeling
- **Cloud Training**: Seamless Kaggle GPU integration for model training
- **Comprehensive Tooling**: Includes annotation tools, model training, and visualization
- **Scalable Architecture**: Designed to handle thousands of product SKUs

## 📁 Project Structure

```
BinSense2/
├── data/                      # Data storage
│   ├── snapshots/             # Training dataset snapshots
│   ├── working/               # Active workspace
│   └── product_embeddings.npz # CLIP embeddings database
│
├── src/                       # Source code
│   ├── clip_classifier_final.py  # CLIP-based product classifier
│   ├── clip_viewer.py         # Streamlit visualization tool
│   ├── diag.py                # Diagnostic tools for data analysis
│   └── templates/             # Web UI templates
│
├── clustered_output/          # Output from clustering pipeline
├── kaggle_output/             # Training outputs and models
│
├── create_combined_snapshot.py # Dataset preparation tool
├── kaggle_deploy_api_combined.py  # Kaggle training deployment
└── kaggle_train_script.py     # Kaggle-compatible training script
```

## 🔧 Core Components

### 1. Data Preparation
- `create_combined_snapshot.py`: Combines existing annotations with clustered product mappings
- `kaggle_train_script.py`: Training script for YOLO model on Kaggle

### 2. Model Training
- `kaggle_deploy_api_combined.py`: Manages Kaggle training jobs
- `src/diag.py`: Diagnostic tools for analyzing clustering results

### 3. Inference & Classification
- `src/clip_classifier_final.py`: CLIP-based product classifier
- `src/clip_viewer.py`: Interactive visualization and evaluation

## 📄 File Descriptions

### Core Scripts (Root Level)

#### Data Preparation & Training
- **`create_combined_snapshot.py`** (15.9KB): Creates training datasets by merging existing annotations with clustered product mappings. Combines multiple data sources into unified training snapshots for YOLO model training.
- **`create_snapshot.py`** (6.3KB): Legacy script for creating dataset snapshots. Used for preparing training data before the combined approach was implemented.
- **`kaggle_train_script.py`** (9.0KB): Main training script optimized for Kaggle environment. Trains YOLOv8 model on product detection dataset with GPU acceleration.
- **`kaggle_deploy_api.py`** (17.1KB): API interface for deploying training jobs to Kaggle. Handles dataset upload, kernel creation, and training job management.
- **`kaggle_deploy_api_combined.py`** (18.6KB): Enhanced version of kaggle_deploy_api with improved dataset handling and training loop integration.

#### Model Evaluation & Testing
- **`check_best.py`** (1.9KB): Utility script to check and validate the best trained model. Performs quick sanity checks on model performance.
- **`final_validation.py`** (3.9KB): Comprehensive validation script that evaluates model performance on held-out test data. Generates detailed metrics and reports.
- **`run_inference.py`** (2.0KB): Simple inference script for running the trained model on individual images. Useful for quick testing and debugging.
- **`view_metrics.py`** (2.9KB): Visualization tool for training metrics. Displays training curves, loss progression, and performance indicators.

#### Configuration & Dependencies
- **`requirements-kaggle.txt`** (222B): Python dependencies for Kaggle training environment. Minimal set required for cloud training.
- **`requirements-local.txt`** (119B): Local development dependencies. Includes additional tools for annotation and visualization.
- **`kernel-metadata.json`** (375B): Kaggle kernel configuration file. Defines environment settings and resource requirements.

#### Data & Metadata
- **`products_by_image.json`** (1.8MB): Mapping of images to their associated products. Contains ground truth annotations linking image IDs to product lists.
- **`readme.md`** (6.2KB): Original project documentation. Contains setup instructions and basic usage examples.

### Source Code (`src/` Directory)

#### CLIP-Based Classification
- **`build_embeddings_from_bins.py`** (12.6KB): Builds CLIP embeddings database from bin images. Extracts visual features from detected objects and creates product embeddings for classification.
- **`clip_classifier_final.py`** (14.9KB): Final CLIP-based product classifier. Implements the two-stage detection pipeline combining YOLO object detection with CLIP-based identification.
- **`clip_classifier.py`** (27.0KB): Legacy CLIP classifier implementation. Earlier version with experimental features and different approach.

#### Visualization & Analysis
- **`clip_viewer.py`** (32.9KB): Interactive Streamlit application for visualizing and evaluating CLIP classifier results. Features include image viewer, statistics dashboard, and performance analytics with random sampling support.
- **`diag.py`** (6.4KB): Diagnostic tools for analyzing clustering results and data statistics. Helps understand data distribution and model performance.

#### Annotation Tools
- **`annotation_app_flask.py`** (30.8KB): Flask-based web application for manual product annotation. Provides browser-based interface for drawing bounding boxes and labeling products.

#### Templates (`src/templates/`)
- Contains HTML templates for the Flask annotation application.

### Training Loop (`training_loop/` Directory)

#### Automated Training Pipeline
- **`deploy.py`** (14.7KB): Main deployment script for automated training loops. Manages the entire training pipeline from dataset preparation to Kaggle deployment.
- **`kaggle_loop.py`** (27.8KB): Core training loop script that runs on Kaggle. Implements iterative training with self-improvement through active learning.
- **`prepare_dataset.py`** (2.4KB): Dataset preparation utilities for the training loop. Handles data preprocessing and validation before training.

#### Documentation
- **`README.md`** (1.8KB): Documentation for the training loop system. Describes the automated training pipeline and usage instructions.

### Product Clustering (`product_clustering/` Directory)

#### Clustering Algorithms
- **`clustering_yoloworld_optimized.py`** (23.5KB): Optimized clustering algorithm using YOLO-World for product grouping. Performs unsupervised clustering of products based on visual similarity.
- **`product_validator.py`** (19.5KB): Validation tools for product clustering results. Ensures clustering quality and provides metrics for cluster evaluation.

#### Pre-trained Models
- **`yolov8x-worldv2.pt`** (146MB): Pre-trained YOLO-World model for open-vocabulary object detection. Used as foundation for product clustering.

### Data Directory Structure

#### Core Data Files
- **`product_embeddings.npz`** (4.2MB): CLIP embeddings database. Contains pre-computed visual features for all products in the catalog.
- **`product_mapping.json`** (139KB): Product ID to metadata mapping. Links product identifiers to names, descriptions, and other attributes.
- **`metrics_history.json`** (6.0KB): Historical training metrics. Tracks model performance over time during training runs.

#### Model Checkpoints
- **`best.pt`** (6.2MB): Best performing YOLO model checkpoint. Used for inference and deployment.
- **`best.pt_bkp`** (6.2MB): Backup of previous best model. Safety copy for rollback if needed.

#### Data Subdirectories
- **`working/`**: Active workspace containing current images, annotations, and metadata
- **`snapshots/`**: Frozen dataset versions used for training reproducibility
- **`deps_output/`**: Output from dependency analysis and processing

### Output Directories

#### Training Outputs
- **`kaggle_output/`**: Results from Kaggle training runs including logs, checkpoints, and performance metrics
- **`clustered_output/`**: Results from product clustering pipeline including cluster assignments and validation metrics

#### Documentation
- **`docs/`**: Project documentation (currently empty)
- **`old_docs/`**: Legacy documentation files
- **`templates/`**: Reusable templates for web interfaces

### Development Environment
- **`venv/`**: Python virtual environment containing all project dependencies
- **`.git/`**: Git repository metadata for version control

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- Kaggle account (for cloud training)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd BinSense2

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🏋️ Training Pipeline

1. **Prepare Dataset**:
   ```bash
   python create_combined_snapshot.py
   ```

2. **Train on Kaggle**:
   ```bash
   python kaggle_deploy_api_combined.py
   ```

3. **Monitor Training**:
   - Track progress in Kaggle notebooks
   - View TensorBoard logs in `kaggle_output/runs/`

## 🔍 Inference and Deployment

### Run CLIP Classifier
```bash
python -m src.clip_classifier_final --test path/to/image.jpg
```

### Launch Visualization Tool
```bash
streamlit run src/clip_viewer.py
```

## 📊 Visualization Tools

### CLIP Viewer
Interactive Streamlit app for visualizing model predictions:
- View detections with confidence scores
- Compare predictions with ground truth
- Analyze model performance metrics

### Diagnostic Tools
- `src/diag.py`: Analyze clustering results and data statistics
- `view_metrics.py`: View training metrics and visualizations

## 🛠️ Technical Stack

### Core Technologies
- **Deep Learning**: PyTorch, Ultralytics YOLO
- **Computer Vision**: OpenCV, CLIP
- **Visualization**: Streamlit, Matplotlib
- **Cloud**: Kaggle API, Google Cloud Storage

### Models
- **Object Detection**: YOLOv8 (custom trained)
- **Feature Extraction**: CLIP (ViT-B/32)
- **Classification**: Custom product classifier

## 📄 License

This project is proprietary and confidential. All rights reserved.

---

*Last updated: January 2024*
