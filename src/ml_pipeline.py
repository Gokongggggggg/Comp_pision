from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATCHING_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "matching" / "all_matching_summary.csv"
FEATURE_STATISTICS_PATH = PROJECT_ROOT / "outputs" / "feature_statistics.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "results"
RESULTS_PATH = RESULTS_DIR / "ml_evaluation_results.csv"
LABEL_MAP = {"real": 0, "fake": 1}
MAX_SVM_TRAIN_ROWS = 5000


def _load_csv_if_exists(csv_path: Path, dataset_name: str) -> pd.DataFrame:
    """Load a CSV file when available, otherwise return an empty dataframe."""
    if not csv_path.exists():
        print(f"Warning: {dataset_name} not found at {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    print(f"Loaded {dataset_name}: {csv_path} ({len(df)} rows)")
    return df


def load_matching_summary() -> pd.DataFrame:
    """Load outputs/matching/all_matching_summary.csv if it exists."""
    return _load_csv_if_exists(MATCHING_SUMMARY_PATH, "matching summary")


def load_feature_statistics() -> pd.DataFrame:
    """Load outputs/feature_statistics.csv if it exists."""
    return _load_csv_if_exists(FEATURE_STATISTICS_PATH, "feature statistics")


def _ensure_image_path(df: pd.DataFrame) -> pd.DataFrame:
    """Create image_path from available path-like columns when needed."""
    df = df.copy()
    if "image_path" in df.columns:
        return df

    for path_column in ("image_b", "augmented_path", "image_a", "original_path", "path"):
        if path_column in df.columns:
            df["image_path"] = df[path_column]
            return df

    df["image_path"] = np.nan
    return df


def _select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the reusable ML feature columns while tolerating missing columns."""
    possible_columns = [
        "method",
        "augmentation",
        "keypoints_original",
        "keypoints_augmented",
        "good_matches",
        "num_keypoints",
        "descriptor_mean",
        "descriptor_variance",
        "image_path",
        "label",
    ]

    for column in possible_columns:
        if column not in df.columns:
            df[column] = np.nan

    return df[possible_columns]


def build_feature_dataset(
    matching_df: pd.DataFrame | None = None,
    feature_stats_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine feature statistics and matching statistics into one dataframe."""
    if matching_df is None:
        matching_df = load_matching_summary()
    if feature_stats_df is None:
        feature_stats_df = load_feature_statistics()

    datasets = []

    if not matching_df.empty:
        matching_features = _ensure_image_path(matching_df)
        if "num_keypoints" not in matching_features.columns:
            matching_features["num_keypoints"] = matching_features.get(
                "keypoints_augmented",
                np.nan,
            )
        datasets.append(_select_feature_columns(matching_features))

    if not feature_stats_df.empty:
        descriptor_features = _ensure_image_path(feature_stats_df)
        if "keypoints_augmented" not in descriptor_features.columns:
            descriptor_features["keypoints_augmented"] = descriptor_features.get(
                "num_keypoints",
                np.nan,
            )
        datasets.append(_select_feature_columns(descriptor_features))

    if not datasets:
        print("Warning: no feature data available to build the ML dataset.")
        return pd.DataFrame()

    combined_df = pd.concat(datasets, ignore_index=True, sort=False)
    print(f"Built combined feature dataset: {len(combined_df)} rows")
    return combined_df


def infer_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Infer real/fake labels from image_path when labels are missing."""
    df = df.copy()
    if "label" not in df.columns:
        df["label"] = np.nan

    if "image_path" not in df.columns:
        print("Warning: image_path column is missing; labels cannot be inferred.")
        return df

    paths = df["image_path"].fillna("").astype(str).str.replace("\\", "/", regex=False)
    inferred_labels = pd.Series(np.nan, index=df.index, dtype="object")
    inferred_labels[paths.str.contains("/real/", case=False, regex=False)] = "real"
    inferred_labels[paths.str.contains("/fake/", case=False, regex=False)] = "fake"

    existing_labels = df["label"].replace({0: "real", 1: "fake", "0": "real", "1": "fake"})
    df["label"] = existing_labels.where(existing_labels.notna(), inferred_labels)

    missing_count = int(df["label"].isna().sum())
    if missing_count:
        print(f"Warning: {missing_count} rows still have missing labels after inference.")

    return df


def preprocess_features(df: pd.DataFrame):
    """Clean rows, encode categorical columns, and return X and y."""
    if df.empty:
        print("Warning: feature dataset is empty.")
        return pd.DataFrame(), pd.Series(dtype=int)

    df = df.copy()
    df = df.dropna(subset=["label"])
    if df.empty:
        print("Warning: all rows were removed because labels are missing.")
        return pd.DataFrame(), pd.Series(dtype=int)

    df["label"] = df["label"].replace({0: "real", 1: "fake", "0": "real", "1": "fake"})
    df = df[df["label"].isin(LABEL_MAP.keys())].copy()
    if df.empty:
        print("Warning: no valid real/fake labels found.")
        return pd.DataFrame(), pd.Series(dtype=int)

    y = df["label"].map(LABEL_MAP).astype(int)
    feature_df = df.drop(columns=["label", "image_path"], errors="ignore")

    categorical_columns = feature_df.select_dtypes(include=["object", "category"]).columns
    feature_df = pd.get_dummies(feature_df, columns=categorical_columns, dummy_na=False)

    for column in feature_df.columns:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")

    X = feature_df.fillna(0)
    print(f"Prepared feature matrix: {X.shape[0]} rows x {X.shape[1]} columns")
    return X, y


def train_random_forest(X_train, y_train):
    """Train a RandomForestClassifier model."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def _sample_svm_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_rows: int = MAX_SVM_TRAIN_ROWS,
):
    """Use a stratified training sample for SVM when the training set is large."""
    if len(X_train) <= max_rows:
        print(f"SVM training rows: {len(X_train)}")
        return X_train, y_train

    X_sample, _, y_sample, _ = train_test_split(
        X_train,
        y_train,
        train_size=max_rows,
        random_state=42,
        stratify=y_train,
    )
    print(f"SVM training rows: {len(X_sample)} (stratified sample from {len(X_train)})")
    return X_sample, y_sample


def train_svm(X_train, y_train):
    """Train an SVM classifier with an RBF kernel."""
    X_svm_train, y_svm_train = _sample_svm_training_data(X_train, y_train)

    model = SVC(
        kernel="rbf",
        probability=True,
        random_state=42,
    )
    model.fit(X_svm_train, y_svm_train)
    return model


def _save_confusion_matrix(cm: np.ndarray, model_name: str) -> Path:
    """Save a confusion matrix figure for a model."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    class_names = ["real", "fake"]
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix - {model_name}",
    )

    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, cm[row, col], ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Evaluate a model and save its confusion matrix figure."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    report = classification_report(
        y_test,
        y_pred,
        target_names=["real", "fake"],
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    confusion_matrix_path = _save_confusion_matrix(cm, model_name)

    print(f"\n{model_name} evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("Classification report:")
    print(report)
    print(f"Saved confusion matrix to: {confusion_matrix_path}")

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
        "confusion_matrix_path": str(confusion_matrix_path),
    }


def save_results(results) -> None:
    """Save model evaluation metrics to CSV."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved ML evaluation results to: {RESULTS_PATH}")


def _can_train_classifier(y: pd.Series) -> bool:
    """Check whether the dataset has enough class diversity for classification."""
    class_counts = y.value_counts()
    if len(class_counts) < 2:
        print("\nWarning: dataset contains only one class.")
        print("Classification is not valid yet because ML needs both real and fake samples.")
        print(f"Current class counts: {class_counts.to_dict()}")
        return False

    if class_counts.min() < 2:
        print("\nWarning: at least one class has fewer than 2 samples.")
        print("Train/test split with reliable evaluation requires more samples per class.")
        print(f"Current class counts: {class_counts.to_dict()}")
        return False

    return True


def main() -> None:
    """Run the reusable classical ML pipeline."""
    matching_df = load_matching_summary()
    feature_stats_df = load_feature_statistics()

    feature_dataset = build_feature_dataset(matching_df, feature_stats_df)
    labeled_dataset = infer_labels(feature_dataset)
    X, y = preprocess_features(labeled_dataset)

    if X.empty or y.empty:
        print("Stopping gracefully because there is no usable ML dataset.")
        return

    if not _can_train_classifier(y):
        return

    stratify = y if y.nunique() == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=stratify,
    )

    random_forest = train_random_forest(X_train, y_train)
    svm = train_svm(X_train, y_train)

    results = [
        evaluate_model(random_forest, X_test, y_test, "random_forest"),
        evaluate_model(svm, X_test, y_test, "svm"),
    ]
    save_results(results)


if __name__ == "__main__":
    main()
