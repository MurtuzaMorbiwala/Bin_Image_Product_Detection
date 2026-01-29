import streamlit as st
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path("C:/Users/bindi/PythonDev/BinSense2")
IMAGES_DIR = BASE_DIR / "data/working/images"
METADATA_DIR = BASE_DIR / "data/working/metadata"
CACHE_DIR = BASE_DIR / "clustered_output/cache"
OUTPUT_DIR = BASE_DIR / "clustered_output"

YOLO_CACHE_FILE = CACHE_DIR / "yolo_detections.json"
CLIP_CACHE_FILE = CACHE_DIR / "clip_embeddings_cache.json"
MAPPING_FILE = OUTPUT_DIR / "product_mapping.json"
SUMMARY_FILE = OUTPUT_DIR / "clustering_summary.json"

# Verify paths
for path, name in [
    (IMAGES_DIR, "Images directory"),
    (METADATA_DIR, "Metadata directory"),
    (YOLO_CACHE_FILE, "YOLO cache"),
    (CLIP_CACHE_FILE, "CLIP cache"),
    (MAPPING_FILE, "Product mapping")
]:
    if not path.exists():
        st.error(f"❌ {name} not found at: {path}")
        st.stop()

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def crop_box_from_image(img_path, bbox_xyxy):
    """Crop a box from an image given xyxy coordinates"""
    try:
        img = Image.open(img_path).convert("RGB")
        x1, y1, x2, y2 = [int(c) for c in bbox_xyxy]
        
        # Ensure valid coordinates
        x1, x2 = max(0, x1), min(img.width, x2)
        y1, y2 = max(0, y1), min(img.height, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
            
        crop = img.crop((x1, y1, x2, y2))
        return crop
    except Exception as e:
        return None

def draw_boxes_on_image(img_path, boxes_to_draw):
    """
    Draw boxes on image.
    
    boxes_to_draw format:
    [
        {'bbox_xyxy': [...], 'color': 'green', 'label': 'Box 1', 'width': 3},
        ...
    ]
    """
    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("arial.ttf", 30)
            small_font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        for box_info in boxes_to_draw:
            x1, y1, x2, y2 = [int(c) for c in box_info['bbox_xyxy']]
            color = box_info.get('color', 'green')
            label = box_info.get('label', '')
            width = box_info.get('width', 5)
            
            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
            
            # Draw label with background
            if label:
                # Calculate text size
                bbox = draw.textbbox((0, 0), label, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Position for text background
                text_x = x1
                text_y = max(0, y1 - text_height - 10)
                
                # Draw background rectangle
                draw.rectangle(
                    [text_x, text_y, text_x + text_width + 10, text_y + text_height + 10],
                    fill=color
                )
                
                # Draw text
                draw.text((text_x + 5, text_y + 5), label, fill="white", font=font)
        
        return img
    except Exception as e:
        st.error(f"Error drawing boxes: {e}")
        return None

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    """Load all necessary data"""
    # Load product mapping (clustering results)
    with open(MAPPING_FILE, 'r') as f:
        product_mapping = json.load(f)
    
    # Load YOLO cache
    with open(YOLO_CACHE_FILE, 'r') as f:
        yolo_cache = json.load(f)
    
    # Load CLIP cache
    with open(CLIP_CACHE_FILE, 'r') as f:
        clip_cache = json.load(f)
    
    # Load summary if available
    summary = None
    if SUMMARY_FILE.exists():
        with open(SUMMARY_FILE, 'r') as f:
            summary = json.load(f)
    
    # Build product index from metadata
    product_to_images = {}
    product_names = {}
    image_to_products = {}
    
    for meta_file in METADATA_DIR.glob("*.json"):
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                bin_data = meta.get("BIN_FCSKU_DATA", {})
                
                img_products = []
                for sku, sku_info in bin_data.items():
                    # Get product name
                    if sku not in product_names:
                        name = (sku_info.get("name") or 
                               sku_info.get("normalizedName") or 
                               f"Product {sku}")
                        product_names[sku] = name
                    
                    # Track images
                    if sku not in product_to_images:
                        product_to_images[sku] = []
                    product_to_images[sku].append(meta_file.stem)
                    img_products.append(sku)
                
                image_to_products[meta_file.stem] = img_products
        except Exception as e:
            continue
    
    return product_mapping, yolo_cache, clip_cache, product_to_images, product_names, image_to_products, summary

# ============================================================
# MAIN APP
# ============================================================
st.set_page_config(layout="wide", page_title="Product Crop Validator")

st.title("🔍 Product Crop Validator")
st.markdown("*First line: All images where product should appear | Second line: Detected crops*")

with st.spinner("Loading data..."):
    product_mapping, yolo_cache, clip_cache, product_to_images, product_names, image_to_products, summary = load_data()

# ============================================================
# BUILD PRODUCT STATISTICS
# ============================================================
product_stats = []

for sku in product_to_images.keys():
    # Count images where product appears
    num_images = len(product_to_images[sku])
    
    # Count detections for this product
    num_detections = 0
    detected_in_images = set()
    
    for img_id, detections in product_mapping.items():
        for det in detections:
            if det.get('original_product_id') == sku:
                num_detections += 1
                detected_in_images.add(img_id)
    
    product_stats.append({
        'sku': sku,
        'name': product_names.get(sku, f"Product {sku}"),
        'num_images': num_images,
        'num_detections': num_detections,
        'num_images_detected': len(detected_in_images),
        'has_detections': num_detections > 0
    })

# ============================================================
# SIDEBAR - FILTERING AND SORTING
# ============================================================
st.sidebar.title("📦 Product Selection")

# Filter by detection status
filter_mode = st.sidebar.radio(
    "Show products:",
    options=["With detections ✓", "Without detections ✗", "All products"],
    index=0
)

# Filter products
if filter_mode == "With detections ✓":
    filtered_stats = [p for p in product_stats if p['has_detections']]
elif filter_mode == "Without detections ✗":
    filtered_stats = [p for p in product_stats if not p['has_detections']]
else:
    filtered_stats = product_stats

# Sort options
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Sort By")

sort_by = st.sidebar.selectbox(
    "Primary sort:",
    options=[
        "Most detections first",
        "Most images first",
        "Detection rate (detections/images)",
        "Alphabetical (SKU)",
        "Alphabetical (Name)"
    ],
    index=0
)

# Sort the products
if sort_by == "Most detections first":
    filtered_stats.sort(key=lambda x: x['num_detections'], reverse=True)
elif sort_by == "Most images first":
    filtered_stats.sort(key=lambda x: x['num_images'], reverse=True)
elif sort_by == "Detection rate (detections/images)":
    filtered_stats.sort(key=lambda x: x['num_detections']/max(x['num_images'], 1), reverse=True)
elif sort_by == "Alphabetical (SKU)":
    filtered_stats.sort(key=lambda x: x['sku'])
elif sort_by == "Alphabetical (Name)":
    filtered_stats.sort(key=lambda x: x['name'])

# Show statistics
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Statistics")

total_products = len(product_stats)
products_with_detections = len([p for p in product_stats if p['has_detections']])

col1, col2 = st.sidebar.columns(2)
col1.metric("Total", total_products)
col2.metric("With ✓", products_with_detections)

success_rate = (products_with_detections / max(total_products, 1)) * 100
st.sidebar.metric("Success Rate", f"{success_rate:.1f}%")

st.sidebar.markdown(f"**Showing:** {len(filtered_stats)} products")

# ============================================================
# PRODUCT SELECTION
# ============================================================
if not filtered_stats:
    st.error("No products match the current filter!")
    st.stop()

# Create product options with statistics
product_options = []
for stat in filtered_stats:
    label = f"{stat['sku']} | Det: {stat['num_detections']} | Imgs: {stat['num_images']} | {stat['name'][:50]}"
    product_options.append((label, stat['sku']))

selected_label = st.sidebar.selectbox(
    "Select product:",
    options=[label for label, _ in product_options],
    index=0
)

selected_sku = next(sku for label, sku in product_options if label == selected_label)

# Get selected product stats
selected_stats = next(p for p in filtered_stats if p['sku'] == selected_sku)

# ============================================================
# DISPLAY SELECTED PRODUCT
# ============================================================
st.markdown("---")

# Header
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
col1.markdown(f"## 🏷️ {selected_sku}")
col2.metric("Images", selected_stats['num_images'])
col3.metric("Detections", selected_stats['num_detections'])
col4.metric("Detected In", f"{selected_stats['num_images_detected']}/{selected_stats['num_images']}")

st.markdown(f"**Product Name:** {selected_stats['name']}")

# Get images where product should appear
expected_images = product_to_images.get(selected_sku, [])

# Get detected crops for this product
detected_crops = []
crops_by_image = defaultdict(list)

for img_id in expected_images:
    detections = product_mapping.get(img_id, [])
    for det in detections:
        if det.get('original_product_id') == selected_sku:
            # Get box embedding
            box_embeddings = clip_cache.get(img_id, [])
            for box_emb in box_embeddings:
                if tuple(box_emb['bbox_norm']) == tuple(det['bbox']):
                    crop_info = {
                        'image_id': img_id,
                        'bbox_xyxy': box_emb['bbox'],
                        'bbox_norm': box_emb['bbox_norm'],
                        'conf': box_emb['conf'],
                        'embedding': box_emb['embedding']
                    }
                    detected_crops.append(crop_info)
                    crops_by_image[img_id].append(crop_info)
                    break

# ============================================================
# IMAGE SIZE CONTROLS
# ============================================================
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    image_height = st.slider("Full image height (px):", 200, 800, 400, 50)

with col2:
    crop_size = st.slider("Crop size (px):", 100, 400, 200, 50)

with col3:
    st.markdown("")  # Spacer

# ============================================================
# FIRST LINE: ALL IMAGES
# ============================================================
st.markdown("---")
st.markdown(f"### 📷 All Images Containing {selected_sku} ({len(expected_images)} images)")

if not expected_images:
    st.warning("No images found for this product in metadata")
else:
    # Calculate number of columns based on image count
    num_cols = min(len(expected_images), 4)
    
    # Display images in rows
    for i in range(0, len(expected_images), num_cols):
        cols = st.columns(num_cols)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(expected_images):
                break
            
            img_id = expected_images[idx]
            img_path = IMAGES_DIR / f"{img_id}.jpg"
            
            with col:
                if img_path.exists():
                    # Check if this image has detections
                    has_detection = img_id in crops_by_image
                    
                    # Draw boxes if detections exist
                    if has_detection:
                        boxes_to_draw = []
                        for crop_idx, crop in enumerate(crops_by_image[img_id]):
                            boxes_to_draw.append({
                                'bbox_xyxy': crop['bbox_xyxy'],
                                'color': 'lime',
                                'label': f'✓ {crop_idx+1}',
                                'width': 6
                            })
                        
                        annotated_img = draw_boxes_on_image(img_path, boxes_to_draw)
                        if annotated_img:
                            # Resize to consistent height
                            aspect_ratio = annotated_img.width / annotated_img.height
                            new_width = int(image_height * aspect_ratio)
                            annotated_img = annotated_img.resize((new_width, image_height), Image.Resampling.LANCZOS)
                            
                            st.image(annotated_img, use_container_width=True)
                            st.success(f"**{img_id}**\n✓ {len(crops_by_image[img_id])} detection(s)")
                    else:
                        # No detection - show original image
                        img = Image.open(img_path)
                        
                        # Resize to consistent height
                        aspect_ratio = img.width / img.height
                        new_width = int(image_height * aspect_ratio)
                        img = img.resize((new_width, image_height), Image.Resampling.LANCZOS)
                        
                        st.image(img, use_container_width=True)
                        st.error(f"**{img_id}**\n✗ No detection")
                else:
                    st.error(f"**{img_id}**\n❌ Image not found")

# ============================================================
# SECOND LINE: DETECTED CROPS
# ============================================================
st.markdown("---")
st.markdown(f"### ✂️ Detected Crops ({len(detected_crops)} total)")

if not detected_crops:
    st.warning(f"No crops detected for {selected_sku}")
    st.info("The clustering algorithm did not find any boxes for this product that met the quality thresholds.")
else:
    # Group crops and show with image context
    st.markdown(f"**Found {len(detected_crops)} crop(s) across {len(crops_by_image)} image(s)**")
    
    # Sort crops by image ID for organized display
    sorted_crops = sorted(detected_crops, key=lambda x: x['image_id'])
    
    # Calculate crops per row (more crops = more columns)
    if len(sorted_crops) <= 5:
        crops_per_row = len(sorted_crops)
    elif len(sorted_crops) <= 10:
        crops_per_row = 5
    elif len(sorted_crops) <= 20:
        crops_per_row = 6
    else:
        crops_per_row = 8
    
    # Display crops in grid
    for i in range(0, len(sorted_crops), crops_per_row):
        cols = st.columns(crops_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(sorted_crops):
                break
            
            crop_info = sorted_crops[idx]
            img_path = IMAGES_DIR / f"{crop_info['image_id']}.jpg"
            
            with col:
                # Crop image
                crop_img = crop_box_from_image(img_path, crop_info['bbox_xyxy'])
                
                if crop_img:
                    # Resize for consistent display
                    crop_img.thumbnail((crop_size, crop_size), Image.Resampling.LANCZOS)
                    st.image(crop_img, use_container_width=True)
                    
                    # Info
                    cx, cy, w, h = crop_info['bbox_norm']
                    area = w * h
                    
                    st.markdown(f"""
                    <div style='text-align: center; font-size: 0.85em; line-height: 1.3;'>
                        <b>{crop_info['image_id']}</b><br>
                        Conf: {crop_info['conf']:.3f}<br>
                        Area: {area:.2%}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("Failed to crop")

# ============================================================
# SUMMARY STATISTICS
# ============================================================
if detected_crops:
    st.markdown("---")
    st.markdown("### 📊 Crop Statistics")
    
    # Calculate stats
    confidences = [c['conf'] for c in detected_crops]
    areas = [c['bbox_norm'][2] * c['bbox_norm'][3] for c in detected_crops]
    
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Total Crops", len(detected_crops))
    col2.metric("Avg Confidence", f"{np.mean(confidences):.3f}")
    col3.metric("Avg Area", f"{np.mean(areas):.2%}")
    col4.metric("Detection Rate", f"{len(crops_by_image)}/{len(expected_images)}")
    
    # Quality indicators
    st.markdown("#### Quality Indicators")
    
    detection_rate = len(crops_by_image) / max(len(expected_images), 1)
    avg_conf = np.mean(confidences)
    
    quality_indicators = []
    
    if detection_rate >= 0.8:
        quality_indicators.append("✅ Good detection rate (≥80%)")
    elif detection_rate >= 0.5:
        quality_indicators.append("⚠️ Moderate detection rate (50-80%)")
    else:
        quality_indicators.append("❌ Low detection rate (<50%)")
    
    if avg_conf >= 0.1:
        quality_indicators.append("✅ Good confidence scores")
    elif avg_conf >= 0.05:
        quality_indicators.append("⚠️ Moderate confidence scores")
    else:
        quality_indicators.append("❌ Low confidence scores")
    
    if len(detected_crops) >= 5:
        quality_indicators.append("✅ Good sample size for learning")
    elif len(detected_crops) >= 3:
        quality_indicators.append("⚠️ Small sample size")
    else:
        quality_indicators.append("❌ Very small sample size")
    
    for indicator in quality_indicators:
        st.markdown(f"- {indicator}")

# ============================================================
# NAVIGATION HINTS
# ============================================================
st.markdown("---")
st.caption("💡 **Tips:** Use the sidebar to sort products and filter by detection status. Adjust image/crop sizes above for better viewing.")
st.caption(f"📦 Currently viewing: {selected_sku} - {selected_stats['name']}")