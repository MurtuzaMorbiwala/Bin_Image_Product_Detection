"""
Improved Streamlit app for CLIP classifier visualization
With statistics, sorting, better analytics, and product name display
"""

import streamlit as st
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys
import os
from collections import defaultdict
import random

# Add src to path if not already there (handles streamlit run from project root)
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import the new classifier
sys.path.insert(0, 'src')
from build_embeddings_from_bins import CLIPClassifier, load_clip_model


# Hardcoded Configuration
IMAGES_DIR = Path(r"data\working\images")
METADATA_DIR = Path(r"data\working\metadata")
YOLO_MODEL = r"kaggle_output\best.pt"
EMBEDDINGS_DB = r"data\product_embeddings.npz"


def load_ground_truth(image_id: str) -> dict:
    """Load ground truth metadata for an image"""
    meta_file = METADATA_DIR / f"{image_id}.json"
    
    if not meta_file.exists():
        return None
    
    try:
        with open(meta_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        st.error(f"Error loading metadata for {image_id}: {e}")
        return None
    
    # Extract product info from BIN_FCSKU_DATA
    products = {}
    if 'BIN_FCSKU_DATA' in metadata:
        for product_id, product_data in metadata['BIN_FCSKU_DATA'].items():
            # The structure has 'name' directly in product_data
            name = product_data.get('name', product_data.get('normalizedName', 'Unknown Product'))
            quantity = product_data.get('quantity', 0)
            
            products[product_id] = {
                'id': product_id,
                'name': name,
                'quantity': quantity,
                'asin': product_data.get('asin', product_id)
            }
    
    return products


def get_product_name(product_id: str, ground_truth: dict) -> str:
    """Get product name from ground truth, fallback to ID"""
    if ground_truth and product_id in ground_truth:
        name = ground_truth[product_id]['name']
        # Truncate very long names
        if len(name) > 40:
            name = name[:37] + "..."
        return f"{name} ({product_id[:8]}...)"
    return f"{product_id[:12]}..."


def draw_bbox_on_image(image: Image.Image, bbox: list, label: str, color: str = "red"):
    """Draw bounding box on image with label"""
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    
    # Draw label background
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
    
    # Truncate label if too long
    max_label_length = 50
    if len(label) > max_label_length:
        label = label[:max_label_length-3] + "..."
    
    # Get text size
    bbox_text = draw.textbbox((x1, y1), label, font=font)
    text_width = bbox_text[2] - bbox_text[0]
    text_height = bbox_text[3] - bbox_text[1]
    
    # Draw background rectangle for text
    draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 8, y1], fill=color)
    draw.text((x1 + 4, y1 - text_height - 2), label, fill="white", font=font)
    
    return image


@st.cache_resource
def load_classifier():
    """Load CLIP classifier (cached)"""
    # Load CLIP model first
    load_clip_model()
    
    classifier = CLIPClassifier(yolo_model_path=YOLO_MODEL)
    classifier.load_database(EMBEDDINGS_DB)
    return classifier


def analyze_all_images(conf_threshold=0.25, sample_size=None, random_seed=42):
    """Analyze images and compute statistics - NOT CACHED
    
    Args:
        conf_threshold: YOLO confidence threshold
        sample_size: Number of images to randomly sample (None = all images)
        random_seed: Random seed for reproducibility
    """
    classifier = load_classifier()
    
    all_image_files = sorted(list(IMAGES_DIR.glob("*.jpg")))
    
    # Random sampling support
    if sample_size and sample_size < len(all_image_files):
        random.seed(random_seed)
        image_files = random.sample(all_image_files, sample_size)
    else:
        image_files = all_image_files
    
    results = []
    overall_stats = {
        'total_images_in_dataset': len(all_image_files),
        'images_analyzed': len(image_files),
        'is_sampled': sample_size is not None and sample_size < len(all_image_files),
        'images_with_detections': 0,
        'images_without_detections': 0,
        'images_with_metadata': 0,
        'perfect_matches': 0,
        'partial_matches': 0,
        'no_matches': 0,
        'total_ground_truth_products': 0,
        'total_detected_products': 0,
        'total_correct': 0,
        'total_false_positives': 0,
        'total_missed': 0,
    }
    
    # Add progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for idx, img_file in enumerate(image_files):
        progress_text.text(f"Processing {idx+1}/{len(image_files)}: {img_file.name}")
        
        image_id = img_file.stem
        
        # Load ground truth
        ground_truth = load_ground_truth(image_id)
        
        if ground_truth:
            overall_stats['images_with_metadata'] += 1
            ground_truth_ids = set(ground_truth.keys())
            overall_stats['total_ground_truth_products'] += len(ground_truth_ids)
        else:
            ground_truth_ids = set()
        
        # Run classifier
        try:
            detections = classifier.classify_image(str(img_file), conf_threshold=conf_threshold)
            detected_ids = set(d['product_id'] for d in detections if d['product_id'] != 'unknown')
            
            if detections:
                overall_stats['images_with_detections'] += 1
            else:
                overall_stats['images_without_detections'] += 1
            
            overall_stats['total_detected_products'] += len(detected_ids)
            
            # Calculate metrics
            correct = ground_truth_ids & detected_ids
            missed = ground_truth_ids - detected_ids
            false_positives = detected_ids - ground_truth_ids
            
            overall_stats['total_correct'] += len(correct)
            overall_stats['total_false_positives'] += len(false_positives)
            overall_stats['total_missed'] += len(missed)
            
            # Categorize images
            if ground_truth_ids:
                if len(correct) == len(ground_truth_ids) and len(false_positives) == 0:
                    overall_stats['perfect_matches'] += 1
                    match_type = 'perfect'
                elif len(correct) > 0:
                    overall_stats['partial_matches'] += 1
                    match_type = 'partial'
                else:
                    overall_stats['no_matches'] += 1
                    match_type = 'none'
            else:
                match_type = 'no_metadata'
            
            recall = len(correct) / len(ground_truth_ids) if ground_truth_ids else 0
            precision = len(correct) / len(detected_ids) if detected_ids else 0
            
            # Strip out PIL Image crops to save memory - we'll regenerate when needed
            detections_for_cache = []
            for det in detections:
                det_copy = det.copy()
                det_copy.pop('crop', None)  # Remove the PIL Image to save memory
                detections_for_cache.append(det_copy)
            
            results.append({
                'image_id': image_id,
                'ground_truth_count': len(ground_truth_ids),
                'detected_count': len(detected_ids),
                'correct_count': len(correct),
                'false_positive_count': len(false_positives),
                'missed_count': len(missed),
                'recall': recall,
                'precision': precision,
                'match_type': match_type,
                'has_detections': len(detections) > 0,
                'detections': detections_for_cache,  # Cache detections without crops
            })
        except Exception as e:
            results.append({
                'image_id': image_id,
                'error': str(e),
                'match_type': 'error',
                'has_detections': False,
                'ground_truth_count': 0,
                'detected_count': 0,
                'correct_count': 0,
                'false_positive_count': 0,
                'missed_count': 0,
                'recall': 0,
                'precision': 0,
                'detections': [],  # Empty detections for error cases
            })
        
        progress_bar.progress((idx + 1) / len(image_files))
    
    progress_bar.empty()
    progress_text.empty()
    
    return results, overall_stats


def show_statistics_tab():
    """Show statistics and analytics"""
    st.header("📊 Overall Statistics")
    
    # Sampling controls
    col1, col2 = st.columns([2, 1])
    
    with col1:
        conf_threshold = st.slider(
            "YOLO Confidence Threshold for Analysis",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            key='stats_conf'
        )
    
    with col2:
        use_sampling = st.checkbox("Use random sampling", value=True, 
                                   help="Analyze a random sample of images for faster results")
        if use_sampling:
            sample_size = st.number_input(
                "Sample size",
                min_value=10,
                max_value=1000,
                value=100,
                step=10,
                help="Number of random images to analyze"
            )
            
            # Warning for large sample sizes
            if sample_size > 500:
                st.warning(f"⚠️ Large sample size ({sample_size}) may use significant memory. Consider using 100-300 for faster performance.")
        else:
            sample_size = None
            st.warning("⚠️ Analyzing ALL images may take a very long time and use significant memory!")
    
    # Add a button to trigger analysis
    analyze_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    # Only run analysis if button is clicked
    if analyze_button:
        with st.spinner("Analyzing images..."):
            results, stats = analyze_all_images(conf_threshold, sample_size)
        
        # Store results in session state for Image Viewer tab
        st.session_state['analysis_results'] = results
        st.session_state['analysis_conf_threshold'] = conf_threshold
        st.session_state['analysis_stats'] = stats
        
        # Estimate memory usage
        import sys
        results_size_mb = sys.getsizeof(str(results)) / (1024 * 1024)
        st.caption(f"💾 Approximate results size: {results_size_mb:.2f} MB in memory")
    
    # Use stored results if available
    if 'analysis_results' in st.session_state:
        results = st.session_state['analysis_results']
        stats = st.session_state['analysis_stats']
        conf_threshold = st.session_state['analysis_conf_threshold']
    else:
        st.info("👆 Click 'Run Analysis' button above to start analyzing images")
        st.warning("💡 You need to run analysis first before using the Image Viewer tab")
        return
    
    # Show sampling warning if applicable
    if stats['is_sampled']:
        st.warning(f"⚠️ Statistics based on **{stats['images_analyzed']} randomly sampled images** out of {stats['total_images_in_dataset']} total images")
    else:
        st.info(f"📊 Statistics based on **all {stats['images_analyzed']} images**")
    
    # Overall metrics
    st.subheader("🎯 Overall Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Images", stats['total_images_in_dataset'])
    with col2:
        st.metric("Images Analyzed", stats['images_analyzed'])
    with col3:
        st.metric("Images with Metadata", stats['images_with_metadata'])
    with col4:
        # Calculate percentage of images where YOLO detected nothing
        no_detection_pct = (stats['images_without_detections'] / stats['images_analyzed'] * 100) if stats['images_analyzed'] > 0 else 0
        st.metric(
            "YOLO No Detections", 
            stats['images_without_detections'],
            delta=f"{no_detection_pct:.1f}% of analyzed",
            delta_color="inverse"
        )
    
    st.divider()
    
    # YOLO Detection Stats
    st.subheader("🔍 YOLO Detection Statistics")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        detection_rate = (stats['images_with_detections'] / stats['images_analyzed'] * 100) if stats['images_analyzed'] > 0 else 0
        st.metric(
            "Images with Detections", 
            stats['images_with_detections'],
            delta=f"{detection_rate:.1f}%",
            help="Images where YOLO detected at least one object"
        )
    with col2:
        st.metric(
            "Images WITHOUT Detections", 
            stats['images_without_detections'],
            delta=f"{no_detection_pct:.1f}%",
            delta_color="inverse",
            help="Images where YOLO detected NOTHING"
        )
    with col3:
        avg_detections = stats['total_detected_products'] / stats['images_with_detections'] if stats['images_with_detections'] > 0 else 0
        st.metric(
            "Avg Detections per Image",
            f"{avg_detections:.1f}",
            help="Average number of products detected per image (for images with detections)"
        )
    
    st.divider()
    
    # Classification Performance
    st.subheader("✅ Classification Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Perfect Matches", stats['perfect_matches'], 
                  help="All products detected correctly, no false positives")
    with col2:
        st.metric("Partial Matches", stats['partial_matches'],
                  help="Some products detected correctly")
    with col3:
        st.metric("No Matches", stats['no_matches'],
                  help="No correct detections")
    
    st.divider()
    
    # Product-level metrics
    st.subheader("📦 Product-Level Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Ground Truth Products", stats['total_ground_truth_products'])
    with col2:
        st.metric("✅ Correct Detections", stats['total_correct'])
    with col3:
        st.metric("❌ Missed Products", stats['total_missed'])
    with col4:
        st.metric("⚠️ False Positives", stats['total_false_positives'])
    
    # Overall recall and precision
    overall_recall = stats['total_correct'] / stats['total_ground_truth_products'] if stats['total_ground_truth_products'] > 0 else 0
    overall_precision = stats['total_correct'] / stats['total_detected_products'] if stats['total_detected_products'] > 0 else 0
    f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Overall Recall", f"{overall_recall:.1%}",
                  help="% of ground truth products correctly detected")
    with col2:
        st.metric("Overall Precision", f"{overall_precision:.1%}",
                  help="% of detections that are correct")
    with col3:
        st.metric("F1 Score", f"{f1_score:.1%}",
                  help="Harmonic mean of precision and recall")
    
    st.divider()
    
    # Breakdown by detection status
    st.subheader("🔍 Detection Status Breakdown")
    
    # Images where YOLO detected nothing
    no_detection_images = [r for r in results if not r['has_detections']]
    
    if no_detection_images:
        st.error(f"**❌ {len(no_detection_images)} images ({no_detection_pct:.1f}%) where YOLO detected NOTHING**")
        
        if st.checkbox("Show images with no detections"):
            for r in no_detection_images[:20]:  # Show first 20
                st.text(f"  - {r['image_id']}")
            if len(no_detection_images) > 20:
                st.text(f"  ... and {len(no_detection_images) - 20} more")
    else:
        st.success("✅ YOLO detected objects in all analyzed images!")
    
    # Detailed breakdown table
    st.subheader("📋 Detailed Results")
    
    # Create DataFrame for display
    import pandas as pd
    df = pd.DataFrame(results)
    
    # Filter options
    filter_type = st.selectbox(
        "Filter by:",
        ["All Images", "Perfect Matches", "Partial Matches", "No Matches", "No Detections", "Has Detections"]
    )
    
    if filter_type == "Perfect Matches":
        df = df[df['match_type'] == 'perfect']
    elif filter_type == "Partial Matches":
        df = df[df['match_type'] == 'partial']
    elif filter_type == "No Matches":
        df = df[df['match_type'] == 'none']
    elif filter_type == "No Detections":
        df = df[df['has_detections'] == False]
    elif filter_type == "Has Detections":
        df = df[df['has_detections'] == True]
    
    # Display table
    display_df = df[['image_id', 'ground_truth_count', 'detected_count', 
                     'correct_count', 'false_positive_count', 'missed_count', 
                     'recall', 'precision']].copy()
    display_df['recall'] = display_df['recall'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    display_df['precision'] = display_df['precision'].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download results
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Results as CSV",
        data=csv,
        file_name="clip_classifier_results.csv",
        mime="text/csv"
    )


def show_image_viewer_tab():
    """Show individual image viewer"""
    st.header("🔍 Image Viewer")
    
    # Check if analysis results exist
    if 'analysis_results' not in st.session_state:
        st.warning("⚠️ Please run analysis in the **Statistics & Analytics** tab first to enable sorting by performance metrics.")
        st.info("👈 Go to the Statistics tab and click analyze to generate results, then come back here.")
        return
    
    # Use analysis results for sorting
    results = st.session_state['analysis_results']
    conf_threshold = st.session_state.get('analysis_conf_threshold', 0.25)
    
    # Sidebar - Sorting options
    st.sidebar.header("📊 Sort Images")
    
    sort_by = st.sidebar.selectbox(
        "Sort by:",
        ["Most Ground Truth Products",
         "Most Detected Products",
         "Most Correct Products",
         "Least Correct Products",
         "Most False Positives",
         "Perfect Matches Only",
         "Partial Matches Only",
         "No Matches Only",
         "No Detections Only",
         "Best Performance (High Correct, Low FP)",
         "Alphabetical"]
    )
    
    # Apply sorting - Create a fresh copy each time
    results_to_sort = list(results)  # Create a copy
    
    if sort_by == "Most Ground Truth Products":
        results_sorted = sorted(results_to_sort, key=lambda x: x.get('ground_truth_count', 0), reverse=True)
    elif sort_by == "Most Detected Products":
        results_sorted = sorted(results_to_sort, key=lambda x: x.get('detected_count', 0), reverse=True)
    elif sort_by == "Most Correct Products":
        results_sorted = sorted(results_to_sort, key=lambda x: x.get('correct_count', 0), reverse=True)
    elif sort_by == "Least Correct Products":
        results_sorted = sorted(results_to_sort, key=lambda x: x.get('correct_count', 0))
    elif sort_by == "Most False Positives":
        results_sorted = sorted(results_to_sort, key=lambda x: x.get('false_positive_count', 0), reverse=True)
    elif sort_by == "Perfect Matches Only":
        results_sorted = [r for r in results_to_sort if r.get('match_type') == 'perfect']
        results_sorted = sorted(results_sorted, key=lambda x: x.get('correct_count', 0), reverse=True)
    elif sort_by == "Partial Matches Only":
        results_sorted = [r for r in results_to_sort if r.get('match_type') == 'partial']
        results_sorted = sorted(results_sorted, key=lambda x: x.get('correct_count', 0), reverse=True)
    elif sort_by == "No Matches Only":
        results_sorted = [r for r in results_to_sort if r.get('match_type') == 'none']
    elif sort_by == "No Detections Only":
        results_sorted = [r for r in results_to_sort if not r.get('has_detections', True)]
    elif sort_by == "Best Performance (High Correct, Low FP)":
        results_sorted = sorted(results_to_sort, key=lambda x: (-x.get('correct_count', 0), x.get('false_positive_count', 0)))
    elif sort_by == "Alphabetical":
        results_sorted = sorted(results_to_sort, key=lambda x: x.get('image_id', ''))
    
    # Create display options for dropdown with product counts
    image_display_options = []
    image_id_to_result = {}
    
    for r in results_sorted:
        image_id = r['image_id']
        gt_count = r.get('ground_truth_count', 0)
        det_count = r.get('detected_count', 0)
        correct_count = r.get('correct_count', 0)
        
        # Format: "image_id [GT:X | Det:Y | ✅Z]"
        display_text = f"{image_id} [GT:{gt_count} | Det:{det_count} | ✅{correct_count}]"
        image_display_options.append(display_text)
        image_id_to_result[display_text] = r
    
    # Show stats for current sorting
    if results_sorted:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Showing {len(results_sorted)} sampled images**")
        
        # Show top 10 for relevant sorts
        if sort_by in ["Most Ground Truth Products", "Most Detected Products", "Most Correct Products", "Most False Positives", "Best Performance (High Correct, Low FP)"]:
            st.sidebar.markdown("**Top 10:**")
            for i, r in enumerate(results_sorted[:10], 1):
                gt = r.get('ground_truth_count', 0)
                det = r.get('detected_count', 0)
                correct = r.get('correct_count', 0)
                fp = r.get('false_positive_count', 0)
                st.sidebar.text(f"{i}. {r['image_id']}")
                st.sidebar.text(f"   GT:{gt} Det:{det} ✅{correct} ❌{fp}")
    
    # Sidebar - Image selection
    st.sidebar.header("📁 Select Image")
    
    if not image_display_options:
        st.error("No images found matching the current filter.")
        return
    
    # Image selector with product counts
    selected_display_option = st.sidebar.selectbox(
        "Choose an image:",
        image_display_options,
        index=0
    )
    
    # Get the actual image_id from the selected display option
    selected_result = image_id_to_result[selected_display_option]
    selected_image_name = selected_result['image_id']
    
    # Show position in sorted list
    position = image_display_options.index(selected_display_option) + 1
    st.sidebar.info(f"Image {position} of {len(image_display_options)}")
    
    selected_image_path = IMAGES_DIR / f"{selected_image_name}.jpg"
    
    # Settings
    st.sidebar.header("⚙️ Display Settings")
    
    # Allow override of confidence threshold for THIS IMAGE ONLY
    viewer_conf_threshold = st.sidebar.slider(
        "YOLO Confidence Threshold (for this image only)",
        min_value=0.1,
        max_value=1.0,
        value=conf_threshold,
        step=0.05,
        key='viewer_conf',
        help="Adjust threshold for viewing this specific image (does not reanalyze all images)"
    )
    
    show_crops = st.sidebar.checkbox("Show cropped detections", value=True)
    show_annotated = st.sidebar.checkbox("Show annotated image", value=True)
    show_metadata_debug = st.sidebar.checkbox("Show metadata debug info", value=False)
    
    # Load ground truth
    st.subheader("📋 Ground Truth (from metadata)")
    ground_truth = load_ground_truth(selected_image_name)
    
    # Debug: Show raw metadata if requested
    if show_metadata_debug and ground_truth:
        with st.expander("🔍 Debug: Raw Metadata"):
            meta_file = METADATA_DIR / f"{selected_image_name}.json"
            with open(meta_file, 'r') as f:
                raw_metadata = json.load(f)
            st.json(raw_metadata)
    
    if ground_truth:
        st.success(f"Found {len(ground_truth)} products in metadata")
        
        # Display ground truth in columns
        num_cols = min(len(ground_truth), 3)
        cols = st.columns(num_cols)
        for idx, (product_id, product_info) in enumerate(ground_truth.items()):
            col_idx = idx % num_cols
            with cols[col_idx]:
                st.markdown(f"""
                **Product {idx + 1}**
                - **ID:** `{product_id}`
                - **Name:** {product_info['name']}
                - **Quantity:** {product_info.get('quantity', 'N/A')}
                """)
    else:
        st.warning("No ground truth metadata found for this image")
        ground_truth = {}
    
    st.divider()
    
    # Run CLIP classifier
    st.subheader("🤖 CLIP Classifier Results")
    
    # Check if we can use cached detections or need to re-run
    # Use cached if threshold matches, otherwise re-run
    use_cached = (viewer_conf_threshold == conf_threshold) and 'detections' in selected_result
    
    if use_cached:
        st.success("✅ Using cached detections from analysis")
        detections = selected_result['detections']
        
        # Cached detections don't have crops - regenerate them if needed for display
        if show_crops and detections:
            # Load the image once
            image = Image.open(selected_image_path).convert('RGB')
            
            # Add crops back to detections
            for det in detections:
                if 'crop' not in det and 'bbox' in det:
                    x1, y1, x2, y2 = det['bbox']
                    det['crop'] = image.crop((x1, y1, x2, y2))
    else:
        if viewer_conf_threshold != conf_threshold:
            st.info(f"⚙️ Re-running classifier with confidence threshold {viewer_conf_threshold:.2f} (original analysis used {conf_threshold:.2f})")
        
        with st.spinner("Running CLIP classifier..."):
            try:
                classifier = load_classifier()
                detections = classifier.classify_image(
                    str(selected_image_path),
                    conf_threshold=viewer_conf_threshold
                )
            except Exception as e:
                st.error(f"Error running classifier: {e}")
                import traceback
                st.code(traceback.format_exc())
                return
    
    # Rest of the detection processing
    try:
        if not detections:
            st.error("❌ No products detected by YOLO")
            st.info("Try lowering the confidence threshold or check if the image contains detectable products")
            return
        
        st.success(f"Detected {len(detections)} products")
        
        # Compare with ground truth
        ground_truth_ids = set(ground_truth.keys())
        detected_ids = set(d['product_id'] for d in detections if d['product_id'] != 'unknown')
        
        correct = ground_truth_ids & detected_ids
        missed = ground_truth_ids - detected_ids
        false_positives = detected_ids - ground_truth_ids
        
        # Show metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ Correct", len(correct))
        with col2:
            st.metric("❌ Missed", len(missed))
        with col3:
            st.metric("⚠️ False Positives", len(false_positives))
        with col4:
            if len(ground_truth_ids) > 0:
                accuracy = len(correct) / len(ground_truth_ids) * 100
                st.metric("📊 Recall", f"{accuracy:.1f}%")
        
        # Show missed products
        if missed:
            with st.expander(f"❌ Missed Products ({len(missed)})", expanded=False):
                for product_id in missed:
                    name = ground_truth[product_id]['name']
                    st.markdown(f"- **{name}** (`{product_id}`)")
        
        # Show annotated image
        if show_annotated:
            st.subheader("📸 Annotated Image")
            
            image = Image.open(selected_image_path).convert('RGB')
            annotated_image = image.copy()
            
            for idx, det in enumerate(detections):
                # Determine color based on correctness
                if det['product_id'] in ground_truth_ids:
                    color = "green"  # Correct
                else:
                    color = "red"  # False positive
                
                # Use product name if available
                label = get_product_name(det['product_id'], ground_truth)
                annotated_image = draw_bbox_on_image(
                    annotated_image,
                    det['bbox'],
                    label,
                    color
                )
            
            st.image(annotated_image, use_container_width=True, caption="Green = Correct, Red = False Positive")
        
        st.divider()
        
        # Show individual detections
        st.subheader("🔎 Individual Detections")
        
        for idx, det in enumerate(detections):
            is_correct = det['product_id'] in ground_truth_ids
            
            # Get product name for header
            display_name = get_product_name(det['product_id'], ground_truth)
            
            with st.expander(
                f"Detection {idx + 1}: {display_name} "
                f"({'✅ CORRECT' if is_correct else '❌ WRONG'})",
                expanded=False
            ):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if show_crops:
                        st.image(det['crop'], caption="Cropped Detection", use_container_width=True)
                    
                    st.markdown(f"""
                    **Product ID:** `{det['product_id']}`  
                    **YOLO Confidence:** {det['yolo_conf']:.1%}  
                    **CLIP Confidence:** {det['clip_conf']:.1%}  
                    **BBox:** {[int(x) for x in det['bbox']]}
                    """)
                
                with col2:
                    st.markdown("**Top 3 CLIP Matches:**")
                    
                    for rank, (match_id, score) in enumerate(det['top_matches'][:3], 1):
                        is_gt = match_id in ground_truth_ids
                        icon = "✅" if is_gt else "⚪"
                        
                        # Get product name if available
                        name = get_product_name(match_id, ground_truth)
                        
                        st.markdown(f"""
                        {rank}. {icon} **{name}** ({score:.1%})  
                        *Full ID: {match_id}*
                        """)
        
    except Exception as e:
        st.error(f"Error processing detections: {e}")
        import traceback
        st.code(traceback.format_exc())


def main():
    st.set_page_config(
        page_title="CLIP Classifier Viewer",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 CLIP Product Classifier Viewer")
    st.markdown("Analyze and visualize CLIP classifier performance")
    
    # Tabs
    tab1, tab2 = st.tabs(["📊 Statistics & Analytics", "🖼️ Image Viewer"])
    
    with tab1:
        show_statistics_tab()
    
    with tab2:
        show_image_viewer_tab()


if __name__ == "__main__":
    main()