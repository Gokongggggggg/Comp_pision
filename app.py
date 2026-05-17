import cv2
import numpy as np
import pandas as pd
import joblib
import streamlit as st
from pathlib import Path
from PIL import Image, UnidentifiedImageError


METHODS = ("SIFT", "ORB", "AKAZE")
MODEL_PATH = Path("models/random_forest_model.pkl")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.pkl")


@st.cache_resource
def load_model_artifacts():
    if not MODEL_PATH.exists() or not FEATURE_COLUMNS_PATH.exists():
        return None, None, None

    try:
        model = joblib.load(MODEL_PATH)
        feature_columns = list(joblib.load(FEATURE_COLUMNS_PATH))
    except Exception as exc:
        return None, None, exc

    return model, feature_columns, None


def create_detector(method: str):
    if method == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError(
                "SIFT is not available in this OpenCV build. "
                "Try ORB or AKAZE, or install a newer opencv-python package."
            )
        return cv2.SIFT_create()

    if method == "ORB":
        return cv2.ORB_create(nfeatures=1500)

    if method == "AKAZE":
        return cv2.AKAZE_create()

    raise ValueError(f"Unsupported method: {method}")


def load_uploaded_image(uploaded_file) -> Image.Image:
    return Image.open(uploaded_file).convert("RGB")


def detect_features(image_rgb: np.ndarray, method: str):
    gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    detector = create_detector(method)
    keypoints, descriptors = detector.detectAndCompute(gray_image, None)

    keypoint_image = cv2.drawKeypoints(
        image_rgb,
        keypoints,
        None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    return gray_image, keypoints, descriptors, keypoint_image


def extract_feature_statistics(keypoints, descriptors, method):
    if descriptors is None or len(descriptors) == 0:
        return {
            "method": method,
            "num_keypoints": 0,
            "descriptor_mean": 0,
            "descriptor_variance": 0,
            "descriptor_rows": 0,
            "descriptor_cols": 0,
        }

    descriptor_rows = descriptors.shape[0]
    descriptor_cols = descriptors.shape[1] if descriptors.ndim > 1 else 1

    return {
        "method": method,
        "num_keypoints": len(keypoints),
        "descriptor_mean": float(np.mean(descriptors)),
        "descriptor_variance": float(np.var(descriptors)),
        "descriptor_rows": descriptor_rows,
        "descriptor_cols": descriptor_cols,
    }


def build_model_input(feature_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    encoded_features = pd.get_dummies(feature_df)

    for column in feature_columns:
        if column not in encoded_features.columns:
            encoded_features[column] = 0

    return encoded_features[feature_columns]


def get_prediction_confidence(model, prediction, probabilities):
    if probabilities is None or len(probabilities) == 0:
        return None

    probability_row = probabilities[0]
    classes = getattr(model, "classes_", None)

    if classes is not None:
        matching_indexes = np.where(classes == prediction)[0]
        if len(matching_indexes) > 0:
            return float(probability_row[matching_indexes[0]])

    try:
        prediction_index = int(prediction)
    except (TypeError, ValueError):
        return float(np.max(probability_row))

    if 0 <= prediction_index < len(probability_row):
        return float(probability_row[prediction_index])

    return float(np.max(probability_row))


st.set_page_config(
    page_title="Feature Detector Demo",
    layout="wide",
)

st.title("Computer Vision Feature Detector Demo")
st.caption("Upload one image and compare simple local feature detectors.")

with st.sidebar:
    st.header("Controls")
    method = st.selectbox("Feature detection method", METHODS)
    uploaded_file = st.file_uploader(
        "Upload image",
        type=("jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"),
    )

if uploaded_file is None:
    st.info("Upload an image to detect SIFT, ORB, or AKAZE keypoints.")
    st.stop()

try:
    pil_image = load_uploaded_image(uploaded_file)
except UnidentifiedImageError:
    st.error("The uploaded file could not be opened as an image.")
    st.stop()

image_rgb = np.array(pil_image)

try:
    gray_image, keypoints, descriptors, keypoint_image = detect_features(
        image_rgb,
        method,
    )
except (RuntimeError, cv2.error) as exc:
    st.error(str(exc))
    st.stop()

descriptor_shape = descriptors.shape if descriptors is not None else None
feature_statistics = extract_feature_statistics(keypoints, descriptors, method)
feature_df = pd.DataFrame([feature_statistics])

metric_columns = st.columns(3)
metric_columns[0].metric("Method", method)
metric_columns[1].metric("Keypoints", len(keypoints))
metric_columns[2].metric(
    "Descriptor shape",
    str(descriptor_shape) if descriptor_shape is not None else "None",
)

image_columns = st.columns(2)
with image_columns[0]:
    st.subheader("Original Image")
    st.image(image_rgb, use_container_width=True)

with image_columns[1]:
    st.subheader("Keypoint Visualization")
    st.image(keypoint_image, use_container_width=True)

st.subheader("Deepfake Classification")
model, feature_columns, model_error = load_model_artifacts()

if model_error is not None:
    st.error(f"Could not load model files: {model_error}")
elif model is None or feature_columns is None:
    st.info(
        "Classification model not found. Add "
        "`models/random_forest_model.pkl` and `models/feature_columns.pkl` "
        "to enable real/fake prediction."
    )
else:
    X = build_model_input(feature_df, feature_columns)
    prediction = model.predict(X)[0]
    prediction_proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    confidence = get_prediction_confidence(model, prediction, prediction_proba)

    if prediction == 1:
        st.error("Prediction: FAKE")
    else:
        st.success("Prediction: REAL")

    if confidence is not None:
        st.metric("Confidence", f"{confidence:.2%}")

st.info(
    "This prediction is based on handcrafted Computer Vision descriptors "
    "(SIFT, ORB, AKAZE) and Random Forest classification. This is an "
    "academic baseline demo, not a production-grade deepfake detector."
)

with st.expander("Extracted Feature Statistics"):
    st.dataframe(feature_df, use_container_width=True)

with st.expander("Grayscale image"):
    st.image(gray_image, clamp=True, use_container_width=True)
