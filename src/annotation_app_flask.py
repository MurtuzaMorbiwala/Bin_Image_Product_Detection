#!/usr/bin/env python3
"""
Flask-based annotation app with HTML5 Canvas for mouse drawing
Browser-based with real click-and-drag bounding box drawing
"""

import os
import json
import glob
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, jsonify, request, send_from_directory
import base64

app = Flask(__name__)

# Configuration
DATA_DIR = "data/working"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
METADATA_DIR = os.path.join(DATA_DIR, "metadata")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")

# Create directories if they don't exist
os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

def load_products_by_image():
    """Load products by image with class IDs included"""
    products_file = "products_by_image.json"
    
    if not os.path.exists(products_file):
        return None
    
    with open(products_file, 'r') as f:
        return json.load(f)

def load_annotations(image_id):
    """Load existing annotations for an image"""
    annotation_file = os.path.join(ANNOTATIONS_DIR, f"{image_id}.txt")
    annotations = []
    
    if os.path.exists(annotation_file):
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    annotations.append({
                        'class_id': int(parts[0]),
                        'x_center': float(parts[1]),
                        'y_center': float(parts[2]),
                        'width': float(parts[3]),
                        'height': float(parts[4])
                    })
    
    return annotations

def save_annotations(image_id, annotations):
    """Save annotations to YOLO format file (class 0 for all products)
    Also saves product mapping for CLIP stage"""
    import json
    
    annotation_file = os.path.join(ANNOTATIONS_DIR, f"{image_id}.txt")
    
    # Save YOLO format with class 0 (single class: "product")
    with open(annotation_file, 'w') as f:
        for ann in annotations:
            # Clamp values to valid range [0, 1]
            x = max(0.0, min(1.0, ann['x_center']))
            y = max(0.0, min(1.0, ann['y_center']))
            w = max(0.0, min(1.0, ann['width']))
            h = max(0.0, min(1.0, ann['height']))
            # Always use class 0 for YOLO training
            f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    
    # Save product mapping for CLIP (preserves original product IDs)
    mapping_file = os.path.join(os.path.dirname(ANNOTATIONS_DIR), 'product_mapping.json')
    
    # Load existing mapping
    product_mapping = {}
    if os.path.exists(mapping_file):
        with open(mapping_file, 'r') as f:
            product_mapping = json.load(f)
    
    # Update mapping for this image
    product_mapping[image_id] = [
        {
            'original_product_id': str(ann['class_id']),
            'bbox': [ann['x_center'], ann['y_center'], ann['width'], ann['height']],
            'source': 'human',  # Mark as human-annotated
            'verified': True     # Human annotations are verified by default
        }
        for ann in annotations
    ]
    
    # Save mapping
    with open(mapping_file, 'w') as f:
        json.dump(product_mapping, f, indent=2)

@app.route('/')
def index():
    """Serve the annotation interface"""
    return render_template('annotate.html')

@app.route('/api/images')
def get_images():
    """Get list of all images with product counts"""
    products_by_image = load_products_by_image()
    if not products_by_image:
        return jsonify({'error': 'Products file not found'}), 404
    
    # Create list with image IDs and product counts
    images_with_counts = [
        {
            'id': img_id,
            'product_count': len(products)
        }
        for img_id, products in products_by_image.items()
    ]
    
    return jsonify({'images': images_with_counts})

@app.route('/api/image/<image_id>')
def get_image(image_id):
    """Get image file"""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        filename = f"{image_id}{ext}"
        image_path = os.path.join(IMAGES_DIR, filename)
        if os.path.exists(image_path):
            # Get absolute path to images directory
            abs_images_dir = os.path.abspath(IMAGES_DIR)
            return send_from_directory(abs_images_dir, filename)
    
    return jsonify({'error': 'Image not found'}), 404

@app.route('/api/products/<image_id>')
def get_products(image_id):
    """Get products for an image"""
    products_by_image = load_products_by_image()
    if not products_by_image:
        return jsonify({'error': 'Products file not found'}), 404
    
    products = products_by_image.get(image_id, [])
    return jsonify({'products': products})

@app.route('/api/annotations/<image_id>')
def get_annotations(image_id):
    annotations = load_annotations(image_id)
    return jsonify({'annotations': annotations})

@app.route('/api/annotations/<image_id>', methods=['POST'])
def save_annotation(image_id):
    """Save a new annotation (adds to existing, doesn't replace)"""
    data = request.json
    
    # Load existing annotations
    annotations = load_annotations(image_id)
    
    # Add new annotation (don't remove existing ones for this product)
    annotations.append({
        'class_id': data['class_id'],
        'x_center': data['x_center'],
        'y_center': data['y_center'],
        'width': data['width'],
        'height': data['height']
    })
    
    # Save
    save_annotations(image_id, annotations)
    
    return jsonify({'success': True})

@app.route('/api/annotations/<image_id>/<int:class_id>', methods=['DELETE'])
def delete_all_annotations_for_product(image_id, class_id):
    """Delete ALL annotations for a specific product"""
    annotations = load_annotations(image_id)
    annotations = [a for a in annotations if a['class_id'] != class_id]
    save_annotations(image_id, annotations)
    
    return jsonify({'success': True})

@app.route('/api/annotations/<image_id>/<int:class_id>/<int:box_index>', methods=['DELETE'])
def delete_single_annotation(image_id, class_id, box_index):
    """Delete a specific bounding box for a product"""
    annotations = load_annotations(image_id)
    
    # Find all boxes for this class
    class_boxes = [i for i, a in enumerate(annotations) if a['class_id'] == class_id]
    
    # Delete the specific box by index
    if box_index < len(class_boxes):
        del annotations[class_boxes[box_index]]
        save_annotations(image_id, annotations)
        return jsonify({'success': True})
    
    return jsonify({'error': 'Box index out of range'}), 400

if __name__ == '__main__':
    # Create templates directory in the same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    templates_dir = os.path.join(script_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    # Update Flask to use the correct template folder
    app.template_folder = templates_dir
    
    # Create the HTML template
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <title>Annotation Tool</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            margin-top: 0;
            color: #333;
        }
        .controls {
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
        }
        .control-group {
            flex: 1;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #555;
        }
        select, button {
            width: 100%;
            padding: 10px;
            font-size: 14px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            font-weight: bold;
        }
        button:hover {
            background: #45a049;
        }
        button.delete {
            background: #f44336;
        }
        button.delete:hover {
            background: #da190b;
        }
        .canvas-container {
            position: relative;
            display: inline-block;
            margin-top: 20px;
        }
        canvas {
            border: 2px solid #ddd;
            cursor: crosshair;
            display: block;
        }
        .info {
            margin-top: 20px;
            padding: 15px;
            background: #e3f2fd;
            border-radius: 4px;
            color: #1976d2;
        }
        .product-list {
            margin-top: 10px;
        }
        .product-item {
            padding: 8px;
            margin: 5px 0;
            background: #f9f9f9;
            border-left: 3px solid #4CAF50;
            cursor: pointer;
        }
        .product-item:hover {
            background: #e8f5e9;
        }
        .product-item.selected {
            background: #c8e6c9;
            border-left-color: #2e7d32;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏷️ Annotation Tool - Click & Drag to Draw</h1>
        
        <div class="controls">
            <div class="control-group">
                <label>Filter by Products:</label>
                <select id="productCountFilter" onchange="filterImages()">
                    <option value="all">All Images</option>
                    <option value="1">1 Product (Single-item bins)</option>
                    <option value="2">2 Products</option>
                    <option value="3">3 Products</option>
                    <option value="4">4 Products</option>
                    <option value="5">5+ Products</option>
                </select>
            </div>
            <div class="control-group">
                <label>Select Image:</label>
                <select id="imageSelect"></select>
                <span id="imageInfo" style="margin-left: 10px; color: #666;"></span>
            </div>
        </div>
        
        <div class="info">
            <strong>Instructions:</strong> Select a product, then <strong>click and drag</strong> to draw bounding boxes. Draw multiple boxes for products with quantity > 1.
            <br><strong>Keyboard Shortcuts:</strong> S=Save | D=Delete Selected Box | Shift+D=Delete All Boxes for Product | N=Next | P=Previous
            <br><strong>To Delete:</strong> Click on a box to select it (turns red), then press D to delete that specific box
        </div>
        
        <div class="product-list" id="productList"></div>
        
        <div class="canvas-container">
            <canvas id="canvas"></canvas>
        </div>
    </div>

    <script>
        let canvas, ctx;
        let currentImage = null;
        let currentImageId = null;
        let currentProduct = null;
        let products = [];
        let annotations = [];
        let allImages = [];  // Store all images with counts
        let filteredImages = [];  // Store filtered images
        
        let isDrawing = false;
        let startX, startY;
        let currentBox = null;
        let selectedBoxIndex = null;  // Track which box is selected for deletion
        
        // Initialize
        window.onload = function() {
            canvas = document.getElementById('canvas');
            ctx = canvas.getContext('2d');
            
            // Mouse events
            canvas.addEventListener('mousedown', startDrawing);
            canvas.addEventListener('mousemove', draw);
            canvas.addEventListener('mouseup', stopDrawing);
            
            // Keyboard shortcuts
            document.addEventListener('keydown', function(e) {
                if (e.key === 's' || e.key === 'S') {
                    e.preventDefault();
                    saveAnnotation();
                } else if (e.key === 'd' || e.key === 'D') {
                    e.preventDefault();
                    if (e.shiftKey) {
                        deleteAllBoxesForProduct();
                    } else {
                        deleteSingleBox();
                    }
                } else if (e.key === 'n' || e.key === 'N') {
                    e.preventDefault();
                    nextImage();
                } else if (e.key === 'p' || e.key === 'P') {
                    e.preventDefault();
                    previousImage();
                }
            });
            
            // Load images
            loadImages();
        };
        
        function loadImages() {
            fetch('/api/images')
                .then(r => r.json())
                .then(data => {
                    allImages = data.images;
                    filteredImages = allImages;
                    populateImageSelect();
                });
        }
        
        function filterImages() {
            const filter = document.getElementById('productCountFilter').value;
            
            if (filter === 'all') {
                filteredImages = allImages;
            } else {
                const count = parseInt(filter);
                filteredImages = allImages.filter(img => {
                    if (count === 5) {
                        return img.product_count >= 5;
                    }
                    return img.product_count === count;
                });
            }
            
            populateImageSelect();
            
            // Show filter stats
            console.log(`Filter: ${filter}, Found: ${filteredImages.length} images`);
        }
        
        function populateImageSelect() {
            const select = document.getElementById('imageSelect');
            select.innerHTML = '';
            
            filteredImages.forEach(img => {
                const option = document.createElement('option');
                option.value = img.id;
                option.textContent = `${img.id} (${img.product_count} product${img.product_count !== 1 ? 's' : ''})`;
                select.appendChild(option);
            });
            
            select.onchange = () => {
                const selectedImg = filteredImages.find(img => img.id === select.value);
                updateImageInfo(selectedImg);
                loadImage(select.value);
            };
            
            if (filteredImages.length > 0) {
                updateImageInfo(filteredImages[0]);
                loadImage(filteredImages[0].id);
            } else {
                document.getElementById('imageInfo').textContent = 'No images match filter';
            }
        }
        
        function updateImageInfo(imageData) {
            const info = document.getElementById('imageInfo');
            info.textContent = `📦 ${imageData.product_count} product${imageData.product_count !== 1 ? 's' : ''} in this bin`;
        }
        
        function loadImage(imageId) {
            currentImageId = imageId;
            currentProduct = null;
            currentBox = null;
            
            // Load all data in parallel and redraw when everything is ready
            Promise.all([
                // Load annotations
                fetch(`/api/annotations/${imageId}`)
                    .then(r => r.json())
                    .then(data => {
                        annotations = data.annotations;
                        console.log(`Loaded ${annotations.length} annotations for ${imageId}`);
                    }),
                
                // Load products
                fetch(`/api/products/${imageId}`)
                    .then(r => r.json())
                    .then(data => {
                        products = data.products;
                        console.log(`Loaded ${products.length} products for ${imageId}`);
                        displayProducts();
                    }),
                
                // Load image
                new Promise((resolve) => {
                    const img = new Image();
                    img.onload = function() {
                        canvas.width = img.width;
                        canvas.height = img.height;
                        currentImage = img;
                        resolve();
                    };
                    img.src = `/api/image/${imageId}`;
                })
            ]).then(() => {
                // All data loaded, now redraw
                console.log('All data loaded, redrawing...');
                redraw();
            });
        }
        
        function displayProducts() {
            const list = document.getElementById('productList');
            list.innerHTML = '<strong>Select Product to Annotate:</strong>';
            
            products.forEach((product, idx) => {
                // Count existing boxes for this product
                const boxCount = annotations.filter(ann => ann.class_id === product.class_id).length;
                const quantity = product.quantity || 1;
                
                const div = document.createElement('div');
                div.className = 'product-item';
                div.textContent = `${idx + 1}. ${product.name} [Qty: ${quantity}, Boxes: ${boxCount}/${quantity}]`;
                div.onclick = () => selectProduct(product);
                list.appendChild(div);
            });
        }
        
        function selectProduct(product) {
            currentProduct = product;
            currentBox = null;
            selectedBoxIndex = null;
            
            // Update UI
            document.querySelectorAll('.product-item').forEach(el => {
                el.classList.remove('selected');
            });
            event.target.classList.add('selected');
            
            redraw();
        }
        
        function startDrawing(e) {
            const rect = canvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;
            
            // Check if clicking on an existing box for the current product
            if (currentProduct) {
                const productBoxes = annotations
                    .map((ann, idx) => ({ann, idx}))
                    .filter(({ann}) => ann.class_id === currentProduct.class_id);
                
                for (let i = 0; i < productBoxes.length; i++) {
                    const {ann, idx} = productBoxes[i];
                    const x = (ann.x_center - ann.width / 2) * canvas.width;
                    const y = (ann.y_center - ann.height / 2) * canvas.height;
                    const w = ann.width * canvas.width;
                    const h = ann.height * canvas.height;
                    
                    // Check if click is inside this box
                    if (clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h) {
                        selectedBoxIndex = i;
                        redraw();
                        return;  // Don't start drawing
                    }
                }
            }
            
            if (!currentProduct) {
                alert('Please select a product first!');
                return;
            }
            
            // Start drawing new box
            selectedBoxIndex = null;
            isDrawing = true;
            startX = clickX;
            startY = clickY;
        }
        
        function draw(e) {
            if (!isDrawing) return;
            
            const rect = canvas.getBoundingClientRect();
            const currentX = e.clientX - rect.left;
            const currentY = e.clientY - rect.top;
            
            currentBox = {
                x: Math.min(startX, currentX),
                y: Math.min(startY, currentY),
                width: Math.abs(currentX - startX),
                height: Math.abs(currentY - startY)
            };
            
            redraw();
        }
        
        function stopDrawing() {
            isDrawing = false;
        }
        
        function redraw() {
            if (!currentImage) return;
            
            console.log(`Redrawing: ${annotations.length} annotations, ${products.length} products`);
            
            // Clear and draw image
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(currentImage, 0, 0);
            
            // Draw existing annotations
            const productBoxes = currentProduct ? 
                annotations.map((ann, idx) => ({ann, idx})).filter(({ann}) => ann.class_id === currentProduct.class_id) : 
                [];
            
            annotations.forEach((ann, annIdx) => {
                console.log(`Drawing annotation for class ${ann.class_id}`);
                const x = (ann.x_center - ann.width / 2) * canvas.width;
                const y = (ann.y_center - ann.height / 2) * canvas.height;
                const w = ann.width * canvas.width;
                const h = ann.height * canvas.height;
                
                // Check if this is the selected product
                const isSelectedProduct = currentProduct && ann.class_id === currentProduct.class_id;
                
                // Check if this specific box is selected for deletion
                const boxIndexInProduct = productBoxes.findIndex(({idx}) => idx === annIdx);
                const isSelectedForDeletion = isSelectedProduct && boxIndexInProduct === selectedBoxIndex;
                
                if (isSelectedForDeletion) {
                    // Red for box selected for deletion
                    ctx.strokeStyle = '#FF0000';
                    ctx.lineWidth = 4;
                    ctx.strokeRect(x, y, w, h);
                    
                    // Label INSIDE the box
                    const label = `[${ann.class_id}] ${currentProduct.name.substring(0, 30)} - SELECTED`;
                    ctx.font = 'bold 14px Arial';
                    const labelWidth = ctx.measureText(label).width + 10;
                    ctx.fillStyle = 'rgba(255, 0, 0, 0.9)';
                    ctx.fillRect(x + 5, y + 5, labelWidth, 25);
                    ctx.fillStyle = 'white';
                    ctx.fillText(label, x + 10, y + 22);
                } else if (isSelectedProduct) {
                    // Green for selected product
                    ctx.strokeStyle = '#00FF00';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(x, y, w, h);
                    
                    // Label INSIDE the box with product name and class ID
                    const label = `[${ann.class_id}] ${currentProduct.name.substring(0, 35)}`;
                    ctx.font = 'bold 14px Arial';
                    const labelWidth = ctx.measureText(label).width + 10;
                    ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
                    ctx.fillRect(x + 5, y + 5, labelWidth, 25);
                    ctx.fillStyle = 'white';
                    ctx.fillText(label, x + 10, y + 22);
                } else {
                    // Grey for other products
                    ctx.strokeStyle = '#cccccc';
                    ctx.lineWidth = 2;
                    ctx.strokeRect(x, y, w, h);
                    
                    // Find product name for this annotation
                    const product = products.find(p => p.class_id === ann.class_id);
                    const label = product ? `[${ann.class_id}] ${product.name.substring(0, 30)}` : `Class ${ann.class_id}`;
                    ctx.font = 'bold 13px Arial';
                    const labelWidth = ctx.measureText(label).width + 10;
                    
                    // Label INSIDE the box with semi-transparent background
                    ctx.fillStyle = 'rgba(200, 200, 200, 0.9)';
                    ctx.fillRect(x + 5, y + 5, labelWidth, 22);
                    ctx.fillStyle = '#000';
                    ctx.fillText(label, x + 10, y + 20);
                }
            });
            
            // Draw current box being drawn
            if (currentBox) {
                ctx.strokeStyle = '#00FF00';
                ctx.lineWidth = 3;
                ctx.strokeRect(currentBox.x, currentBox.y, currentBox.width, currentBox.height);
                
                ctx.fillStyle = '#00FF00';
                ctx.fillRect(currentBox.x, currentBox.y - 25, 200, 25);
                ctx.fillStyle = 'white';
                ctx.font = 'bold 14px Arial';
                ctx.fillText('Drawing - Press S to Save', currentBox.x + 5, currentBox.y - 7);
            }
        }
        
        function saveAnnotation() {
            if (!currentBox || !currentProduct) {
                alert('Please draw a bounding box first!');
                return;
            }
            
            // Convert to YOLO format
            const x_center = (currentBox.x + currentBox.width / 2) / canvas.width;
            const y_center = (currentBox.y + currentBox.height / 2) / canvas.height;
            const width = currentBox.width / canvas.width;
            const height = currentBox.height / canvas.height;
            
            fetch(`/api/annotations/${currentImageId}`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    class_id: currentProduct.class_id,
                    x_center: x_center,
                    y_center: y_center,
                    width: width,
                    height: height
                })
            })
            .then(r => r.json())
            .then(() => {
                // Show success message with file path
                const fileName = `data/working/annotations/${currentImageId}.txt`;
                console.log(`✅ Saved annotation to ${fileName}`);
                
                // Visual feedback
                const statusDiv = document.createElement('div');
                statusDiv.style.cssText = 'position: fixed; top: 20px; right: 20px; background: #4CAF50; color: white; padding: 15px 20px; border-radius: 5px; font-weight: bold; z-index: 1000;';
                statusDiv.textContent = `✅ Saved to ${fileName}`;
                document.body.appendChild(statusDiv);
                setTimeout(() => statusDiv.remove(), 3000);
                
                currentBox = null;
                loadImage(currentImageId);
            })
            .catch(err => {
                alert('❌ Error saving annotation: ' + err);
            });
        }
        
        function deleteSingleBox() {
            if (!currentProduct) {
                alert('Please select a product first!');
                return;
            }
            
            if (selectedBoxIndex === null) {
                alert('Please click on a box to select it first!');
                return;
            }
            
            fetch(`/api/annotations/${currentImageId}/${currentProduct.class_id}/${selectedBoxIndex}`, {
                method: 'DELETE'
            })
            .then(r => r.json())
            .then(() => {
                console.log('🗑️ Deleted single box!');
                selectedBoxIndex = null;
                currentBox = null;
                loadImage(currentImageId);
            })
            .catch(err => {
                alert('❌ Error deleting box: ' + err);
            });
        }
        
        function deleteAllBoxesForProduct() {
            if (!currentProduct) {
                alert('Please select a product first!');
                return;
            }
            
            if (!confirm(`Delete ALL ${annotations.filter(a => a.class_id === currentProduct.class_id).length} boxes for this product?`)) {
                return;
            }
            
            fetch(`/api/annotations/${currentImageId}/${currentProduct.class_id}`, {
                method: 'DELETE'
            })
            .then(r => r.json())
            .then(() => {
                console.log('🗑️ Deleted all boxes for product!');
                selectedBoxIndex = null;
                currentBox = null;
                loadImage(currentImageId);
            })
            .catch(err => {
                alert('❌ Error deleting boxes: ' + err);
            });
        }
        
        function nextImage() {
            const select = document.getElementById('imageSelect');
            const currentIndex = select.selectedIndex;
            if (currentIndex < select.options.length - 1) {
                select.selectedIndex = currentIndex + 1;
                loadImage(select.value);
            }
        }
        
        function previousImage() {
            const select = document.getElementById('imageSelect');
            const currentIndex = select.selectedIndex;
            if (currentIndex > 0) {
                select.selectedIndex = currentIndex - 1;
                loadImage(select.value);
            }
        }
    </script>
</body>
</html>'''
    
    html_file = os.path.join(templates_dir, 'annotate.html')
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("\n" + "="*60)
    print("🏷️  ANNOTATION TOOL - Browser Based")
    print("="*60)
    print("\nStarting server...")
    print("Open your browser and go to: http://localhost:5000")
    print("\nControls:")
    print("  🖱️  Click and drag on the image to draw bounding boxes")
    print("  💾 Click 'Save' to save the annotation")
    print("  🗑️  Click 'Delete' to remove annotation")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)
