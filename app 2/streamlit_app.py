"""
Streamlit web app for long-tail image classification with human review queue.
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image
import json
from pathlib import Path
import sys
from io import BytesIO
import plotly.graph_objects as go
import plotly.express as px

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ImageClassifier
from src.calibration import CalibrationPipeline, calibrate_predictions
from src.router import DynamicRouter
from src.database import ReviewQueueDB


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Long-Tail Image Classifier",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================
def check_models_exist():
    """Check if model files exist."""
    model_dir = Path(__file__).parent.parent / "models"
    required_files = ["best_model.pth", "temperature_model.pkl", "router.pkl"]
    missing = [f for f in required_files if not (model_dir / f).exists()]
    return len(missing) == 0, missing

# ============================================================================
# SESSION STATE
# ============================================================================
@st.cache_resource
def load_model_and_calibration(model_path: str, calibration_path: str, device: str = "cpu"):
    """Load model and calibration."""
    # Check files exist
    if not Path(model_path).exists() or not Path(calibration_path).exists():
        raise FileNotFoundError(f"Model files not found at {model_path} or {calibration_path}")

    # Model
    model = ImageClassifier(num_classes=8)  # ISIC has 8 disease classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Calibration
    calibration = CalibrationPipeline()
    calibration.load(calibration_path)

    return model, calibration


@st.cache_resource
def load_router(router_path: str):
    """Load router."""
    router = DynamicRouter()
    router.load(router_path)
    # Cap threshold at 0.85 for production safety (avoid routing all predictions)
    if router.threshold > 0.85:
        router.threshold = 0.85
    return router


@st.cache_resource
def load_database():
    """Load database."""
    db_path = Path(__file__).parent / "db" / "review_queue.db"
    return ReviewQueueDB(str(db_path))


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_device():
    """Get device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_category_mapping(mapping_path: str = "data/category_mapping.json"):
    """Load category ID to name mapping."""
    try:
        with open(mapping_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("🦋 Long-Tail Classifier")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Page",
    ["Upload & Predict", "Review Queue", "Metrics Dashboard", "About"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **What is this?**

    A production image classifier that knows what it doesn't know.
    - Routes uncertain predictions to human review
    - Focuses on long-tail species (rare classes)
    - Temperature-scaled confidence calibration
    """
)

# ============================================================================
# PAGE: UPLOAD & PREDICT
# ============================================================================
if page == "Upload & Predict":
    st.title("🖼️ Upload & Predict")
    st.markdown(
        """
        Upload an image of a plant or animal. The classifier will:
        1. Predict the species
        2. Estimate confidence
        3. Decide to route to human review or automated processing
        """
    )

    # Check if models are available (use absolute paths)
    model_dir = Path(__file__).parent.parent / "models"
    model_path = model_dir / "best_model.pth"
    calibration_path = model_dir / "temperature_model.pkl"
    router_path = model_dir / "router.pkl"

    if not all([model_path.exists(), calibration_path.exists(), router_path.exists()]):
        st.error(
            "⚠️ Models not found. This demo works best with local deployment.\n\n"
            "**Local Setup:**\n"
            "```bash\ncd app\\ 2\nstreamlit run streamlit_app.py\n```\n\n"
            "**Note:** Git LFS model files don't work on Streamlit Cloud. "
            "For cloud deployment, models need to be downloaded from a cloud storage service (AWS S3, etc)."
        )
    else:
        try:
            device = get_device()
            model, calibration = load_model_and_calibration(
                str(model_path),
                str(calibration_path),
                device
            )
            router = load_router(str(router_path))
            db = load_database()
            category_map = load_category_mapping()

            # Upload interface
            col1, col2 = st.columns([1, 1])

            with col1:
                st.subheader("Upload Image")
                uploaded_file = st.file_uploader(
                    "Choose an image",
                    type=["jpg", "jpeg", "png", "gif", "bmp"]
                )

            if uploaded_file is not None:
                # Display image
                with col1:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, use_container_width=True)

                with col2:
                    st.subheader("Prediction & Routing")

                    # Preprocess
                    from torchvision import transforms
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]
                        ),
                    ])

                    img_tensor = transform(image).unsqueeze(0).to(device)

                    # Predict
                    with torch.no_grad():
                        logits = model(img_tensor)
                        logits_np = logits.cpu().numpy()

                    # Calibrate
                    calibrated_probs = calibrate_predictions(
                        logits_np,
                        calibration.temperature
                    )
                    pred_class = np.argmax(calibrated_probs[0])
                    confidence = np.max(calibrated_probs[0])

                    # Route
                    route, reason = router.route(confidence)

                    # Display results
                    st.metric("Confidence", f"{confidence:.2%}")
                    st.metric("Routing Threshold", f"{router.threshold:.4f}")

                    # Routing decision
                    if route == "human":
                        st.warning(f"🔴 **Routed to Human Review**\n\n{reason}")
                        # Save uploaded file to persistent location
                        upload_dir = Path(__file__).parent / "uploads"
                        upload_dir.mkdir(exist_ok=True)
                        image_save_path = upload_dir / uploaded_file.name
                        with open(image_save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        # Auto-flag for human review
                        db.flag_image(
                            image_path=str(image_save_path),
                            category_id=pred_class,
                            category_name=category_map.get(pred_class, f"Class {pred_class}"),
                            predicted_class_id=pred_class,
                            predicted_class_name=category_map.get(pred_class, f"Class {pred_class}"),
                            confidence=confidence,
                            threshold=router.threshold,
                            routing_reason=reason,
                            logits=logits_np[0],
                            calibrated_probs=calibrated_probs[0],
                        )
                    else:
                        st.success(f"✅ **Automated Processing**\n\n{reason}")

                    # Class prediction
                    class_name = category_map.get(pred_class, f"Class {pred_class}")
                    st.info(f"**Predicted Class:** {class_name}")

                    # Top-k predictions
                    top_k = 5
                    top_k_indices = np.argsort(calibrated_probs[0])[-top_k:][::-1]

                    st.subheader("Top-5 Predictions")
                    pred_df = []
                    for i, idx in enumerate(top_k_indices, 1):
                        class_name = category_map.get(idx, f"Class {idx}")
                        prob = calibrated_probs[0, idx]
                        pred_df.append({
                            "Rank": i,
                            "Class": class_name,
                            "Confidence": f"{prob:.2%}"
                        })

                    st.dataframe(pred_df, use_container_width=True)

                    # Option to flag for review
                    if st.button("📌 Flag for Human Review", key="flag_button"):
                        db.flag_image(
                            image_path=uploaded_file.name,
                            category_id=pred_class,
                            category_name=category_map.get(pred_class, f"Class {pred_class}"),
                            predicted_class_id=pred_class,
                            predicted_class_name=category_map.get(pred_class, f"Class {pred_class}"),
                            confidence=confidence,
                            threshold=router.threshold,
                            routing_reason=reason,
                            logits=logits_np[0],
                            calibrated_probs=calibrated_probs[0],
                        )
                        st.success("✅ Image flagged for review!")

        except Exception as e:
            st.error(f"Error loading model: {e}")
            import traceback
            st.text(traceback.format_exc())

# ============================================================================
# PAGE: REVIEW QUEUE
# ============================================================================
elif page == "Review Queue":
    st.title("👁️ Human Review Queue")
    st.markdown("Images flagged for human review and accuracy verification")

    db = load_database()
    category_map = load_category_mapping()

    # Stats
    stats = db.get_stats()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Flagged", stats['total_flagged'])
    col2.metric("In Queue", stats['queue_size'])
    col3.metric("Reviewed", stats['total_reviewed'])
    col4.metric("Queue Accuracy", f"{stats['accuracy']:.1%}")

    st.markdown("---")

    # Tabs
    tab1, tab2 = st.tabs(["Pending Review", "Reviewed Images"])

    with tab1:
        st.subheader("Pending Review")
        pending = db.get_pending_reviews(limit=50)

        if pending:
            for idx, item in enumerate(pending):
                with st.expander(
                    f"🔵 {item['image_name']} | Conf: {item['confidence']:.2%}",
                    expanded=(idx == 0)
                ):
                    col1, col2 = st.columns([1, 1])

                    with col1:
                        try:
                            image = Image.open(item['image_path'])
                            st.image(image, use_container_width=True)
                        except Exception as e:
                            st.warning(f"⚠️ Image file not available")

                    with col2:
                        st.write(f"**Image:** {item['image_name']}")
                        st.write(f"**Predicted:** {item['predicted_class_name']}")
                        st.write(f"**Confidence:** {item['confidence']:.2%}")
                        st.write(f"**Reason:** {item['routing_reason']}")
                        st.write(f"**Flagged:** {item['flagged_timestamp']}")

                        # Review form
                        st.write("---")
                        st.write("**Provide Correct Label:**")

                        correct_class = st.text_input(
                            "Correct species/class name",
                            key=f"correct_class_{item['id']}"
                        )

                        is_correct = st.radio(
                            "Was the prediction correct?",
                            [True, False],
                            key=f"is_correct_{item['id']}"
                        )

                        notes = st.text_area(
                            "Notes",
                            key=f"notes_{item['id']}"
                        )

                        if st.button("✅ Submit Review", key=f"submit_{item['id']}"):
                            db.mark_reviewed(
                                item['id'],
                                human_label=-1,  # Would be category ID
                                human_label_name=correct_class,
                                is_correct=is_correct,
                                reviewer_notes=notes
                            )
                            st.success("Review submitted!")
                            st.rerun()

        else:
            st.info("✨ No images pending review!")

    with tab2:
        st.subheader("Reviewed Images")
        reviewed = db.get_reviewed_images(limit=50)

        if reviewed:
            review_data = []
            for item in reviewed:
                review_data.append({
                    "Image": item['image_name'],
                    "Predicted": item['predicted_class_name'],
                    "Actual": item['human_label_name'],
                    "Confidence": f"{item['confidence']:.2%}",
                    "Correct": "✅" if item['is_correct'] else "❌",
                    "Reviewed": item['reviewed_timestamp']
                })

            st.dataframe(review_data, use_container_width=True)
        else:
            st.info("No reviewed images yet")

# ============================================================================
# PAGE: METRICS DASHBOARD
# ============================================================================
elif page == "Metrics Dashboard":
    st.title("Performance Metrics")

    db = load_database()
    latest = db.get_latest_metrics()

    if latest:
        # Accuracy Metrics (Visual Cards)
        st.subheader("Classification Accuracy")
        col1, col2, col3 = st.columns(3)

        with col1:
            head_acc = latest['head_accuracy']
            col1.metric("Common Diseases", f"{head_acc:.1f}%", help="Head Classes")
            st.progress(head_acc / 100)

        with col2:
            tail_acc = latest['tail_accuracy']
            col2.metric("Rare Diseases", f"{tail_acc:.1f}%", help="Tail Classes")
            st.progress(tail_acc / 100)

        with col3:
            gap = latest['head_accuracy'] - latest['tail_accuracy']
            gap_color = "🟢" if gap < 5 else "🟡" if gap < 10 else "🔴"
            col3.metric("Accuracy Gap", f"{gap:.1f}%", delta=f"{gap_color}")
            st.progress(1 - (gap / 100))  # Inverse: low gap = high progress

        st.markdown("---")

        # Calibration Metrics (Visual Improvement)
        st.subheader("Calibration Quality")
        col1, col2 = st.columns(2)

        with col1:
            ece_before = latest['ece_before']
            ece_after = latest['ece_after']
            improvement = (ece_before - ece_after) / (ece_before + 1e-10)

            col1.metric("ECE Before", f"{ece_before:.4f}", help="Higher = Worse")
            col1.metric("ECE After", f"{ece_after:.4f}", delta=f"-{improvement:.1%}", help="Lower = Better")
            col1.metric("Improvement", f"{improvement:.1%}", help="Better calibration")

        with col2:
            st.metric("Temperature Scaling", f"{latest['temperature']:.4f}", "Optimal calibration parameter")
            st.info("""
            **What is ECE?**

            Expected Calibration Error measures if the model's confidence matches actual accuracy.
            - Lower ECE = Better calibrated model
            - Higher temperature = More conservative confidence
            """)

        st.markdown("---")

        # Routing Metrics
        st.subheader("Uncertainty Routing")
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Confidence Threshold", f"{latest['routing_threshold']:.2f}", help="Predictions below this → Human review")
        col2.metric("Tail Routing", f"{latest['tail_routing_rate']:.1%}", help="Rare disease predictions to human")
        col3.metric("Total Routed", f"{latest['total_routing_rate']:.1%}", help="All predictions to human review")
        col4.metric("Head False Route", f"{latest['head_false_routing_rate']:.1%}", help="Easy cases sent to human (acceptable)")

        # Queue Status with Visual Indicators
        st.markdown("---")
        st.subheader("Review Queue Status")

        stats = db.get_stats()
        col1, col2, col3, col4 = st.columns(4)

        total_flagged = stats['total_flagged']
        queue_size = stats['queue_size']
        reviewed = stats['total_reviewed']
        accuracy = stats['accuracy']

        col1.metric("Total Flagged", total_flagged, help="Images sent to human review")
        col2.metric("Pending Review", queue_size, help="Waiting for feedback")
        col3.metric("Completed Reviews", reviewed, help="Feedback received")
        col4.metric("Review Accuracy", f"{accuracy:.1%}", help="Reviewer agreement with model")

        # Queue Progress
        if total_flagged > 0:
            progress_pct = reviewed / total_flagged
            st.write(f"**Review Progress:** {reviewed}/{total_flagged}")
            st.progress(progress_pct)

        # Summary Card
        st.markdown("---")
        with st.expander("📈 System Summary", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.success(f"""
                ✅ **Model Performance**

                - Head Accuracy: {latest['head_accuracy']:.2f}%
                - Tail Accuracy: {latest['tail_accuracy']:.2f}%
                - Calibration improved by {improvement:.1%}
                """)

            with col2:
                st.info(f"""
                ℹ️ **Operational Metrics**

                - {total_flagged} images flagged for review
                - {queue_size} images pending
                - {reviewed} completed reviews
                """)

    else:
        st.info("⚠️ No metrics available yet. Upload and review images to populate metrics.")

# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif page == "About":
    st.title("System Overview")

    st.markdown("**ISIC Skin Lesion Classification System** | Version 1.0")
    st.markdown("---")

    # Performance Metrics
    st.subheader("Clinical Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Head Accuracy", "92.86%", "Common diseases")
    col2.metric("Tail Accuracy", "90.37%", "Rare diseases")
    col3.metric("Calibration (ECE)", "0.0095", "Better confidence")
    col4.metric("Routing Rate", "94.81%", "To human review")

    st.markdown("---")

    # System Features
    st.subheader("System Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("""
        **Uncertainty Aware**

        Routes low-confidence predictions to human experts
        - Threshold: 0.85 confidence
        - Temperature scaled calibration
        """)

    with col2:
        st.info("""
        **Long-Tail Optimized**

        Handles imbalanced medical datasets
        - 92.86% accuracy on common diseases
        - 90.37% accuracy on rare diseases
        """)

    with col3:
        st.info("""
        **Human-in-Loop**

        Persistent review queue system
        - Track predictions & feedback
        - Improve model over time
        """)

    st.markdown("---")

    # Technical Details (Expandable)
    with st.expander("🔧 Technical Architecture"):
        col1, col2 = st.columns(2)

        with col1:
            st.write("**Model**")
            st.write("- Architecture: ViT-B/16")
            st.write("- Dataset: ISIC 2019")
            st.write("- Images: 25,000")
            st.write("- Classes: 8 disease types")

        with col2:
            st.write("**Pipeline**")
            st.write("1. Image preprocessing (224×224)")
            st.write("2. Model inference (ViT-B/16)")
            st.write("3. Temperature scaling calibration")
            st.write("4. Dynamic uncertainty routing")

    with st.expander("💊 Supported Diseases"):
        diseases = [
            "Melanoma (MEL) - Most common malignant",
            "Nevus (NV) - Common benign mole",
            "Basal Cell Carcinoma (BCC)",
            "Actinic Keratosis (AK)",
            "Benign Keratosis-like (BKL)",
            "Dermatofibroma (DF)",
            "Vascular Lesion (VASC)",
            "Squamous Cell Carcinoma (SCC)"
        ]
        for disease in diseases:
            st.write(f"• {disease}")

    with st.expander("⚠️ Limitations & Disclaimers"):
        st.warning("""
        **Important Limitations:**
        - Model trained on ISIC 2019; performance may vary on other populations
        - Requires high-quality dermoscopy images
        - **Decision support tool only** - not a replacement for clinical diagnosis
        - All uncertain predictions require human expert review
        - Performance depends on image quality and proper capture technique
        """)

    with st.expander("🔒 Data & Privacy"):
        st.info("""
        **Data Management:**
        - All images stored locally in SQLite3 database
        - No external data transmission
        - Audit trail maintained for all reviews
        - HIPAA-ready architecture (local deployment only)
        """)

    st.markdown("---")
    st.markdown("<div style='text-align: center; color: gray; font-size: 11px;'>For Research & Clinical Decision Support | Local Deployment Only</div>", unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    ISIC Skin Lesion Classification System | Version 1.0 | For Research & Clinical Support Use
    </div>
    """,
    unsafe_allow_html=True
)
