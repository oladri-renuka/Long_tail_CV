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
# SESSION STATE
# ============================================================================
@st.cache_resource
def load_model_and_calibration(model_path: str, calibration_path: str, device: str = "cpu"):
    """Load model and calibration."""
    # Model
    model = ImageClassifier(num_classes=10000)  # iNaturalist has 10K species
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
    return router


@st.cache_resource
def load_database():
    """Load database."""
    return ReviewQueueDB()


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

    # Check if models are available
    model_path = Path("models/best_model.pth")
    calibration_path = Path("models/temperature_model.pkl")
    router_path = Path("models/router.pkl")

    if not all([model_path.exists(), calibration_path.exists(), router_path.exists()]):
        st.error(
            "⚠️ Models not found. Please run the training pipeline first:\n"
            "```bash\npython scripts/train.py\npython scripts/calibrate.py\n```"
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
                    st.image(image, use_column_width=True)

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
                            st.image(image, use_column_width=True)
                        except:
                            st.info("Image file not available")

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
    st.title("📊 Metrics Dashboard")

    db = load_database()

    # Latest metrics
    latest = db.get_latest_metrics()

    if latest:
        st.subheader("Latest Model Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Head Accuracy", f"{latest['head_accuracy']:.2%}")
        col2.metric("Tail Accuracy", f"{latest['tail_accuracy']:.2%}")
        col3.metric("Accuracy Gap", f"{latest['head_accuracy'] - latest['tail_accuracy']:.2%}")

        st.markdown("---")

        st.subheader("Calibration Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("ECE Before", f"{latest['ece_before']:.4f}")
        col2.metric("ECE After", f"{latest['ece_after']:.4f}")
        col3.metric("Temperature", f"{latest['temperature']:.4f}")
        col4.metric("ECE Improvement", f"{(latest['ece_before'] - latest['ece_after'])/(latest['ece_before'] + 1e-10):.1%}")

        st.markdown("---")

        st.subheader("Routing Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Threshold", f"{latest['routing_threshold']:.4f}")
        col2.metric("Total Routing Rate", f"{latest['total_routing_rate']:.2%}")
        col3.metric("Tail Routing Rate", f"{latest['tail_routing_rate']:.2%}")
        col4.metric("Head False Routing", f"{latest['head_false_routing_rate']:.2%}")

        # Queue status
        st.markdown("---")
        st.subheader("Queue Status")

        stats = db.get_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Flagged", stats['total_flagged'])
        col2.metric("In Queue", stats['queue_size'])
        col3.metric("Review Accuracy", f"{stats['accuracy']:.1%}")

    else:
        st.info("No metrics available yet. Run the training pipeline first.")

# ============================================================================
# PAGE: ABOUT
# ============================================================================
elif page == "About":
    st.title("ℹ️ About This System")

    st.markdown(
        """
        ## Long-Tail Image Classifier

        ### Problem
        Real-world datasets have **long-tail distributions**:
        - Top 100 species: thousands of training images each
        - Bottom 1000 species: <10 images each

        Traditional classifiers struggle on rare (tail) classes.

        ### Solution
        A production system that:
        1. **Fine-tunes ViT-B/16** on iNaturalist 2021 (500K images, 10K species)
        2. **Calibrates predictions** using temperature scaling
        3. **Routes uncertain predictions** to human review
        4. **Maintains a review queue** for accuracy improvement

        ### Key Features
        - ✅ Knows what it doesn't know (uncertainty estimation)
        - ✅ Dynamic routing threshold (95% of tail class predictions to human)
        - ✅ Expected Calibration Error (ECE) metric
        - ✅ Head vs tail accuracy tracking
        - ✅ Cost-benefit analysis

        ### Technology Stack
        - **Model**: Vision Transformer (ViT-B/16) from timm
        - **Calibration**: Temperature scaling + netcal ECE
        - **Database**: SQLite3
        - **Frontend**: Streamlit
        - **Dataset**: iNaturalist 2021 mini

        ### Metrics
        | Metric | Purpose |
        |--------|---------|
        | **ECE** | Measures calibration quality (confidence vs accuracy) |
        | **Head Accuracy** | Performance on common species |
        | **Tail Accuracy** | Performance on rare species |
        | **Routing Rate** | % of predictions sent to human |
        | **False Routing** | % of easy predictions incorrectly routed |

        ### Workflow
        1. Upload any plant/animal photo
        2. Get instant prediction + confidence score
        3. System routes to human if uncertain
        4. Human reviewer provides feedback
        5. Metrics dashboard tracks improvement

        ---

        Built with ❤️ for production robustness and interpretability.
        """
    )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    🦋 Long-Tail Image Classifier | Powered by ViT-B/16 + Temperature Scaling
    </div>
    """,
    unsafe_allow_html=True
)
