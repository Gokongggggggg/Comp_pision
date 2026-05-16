import csv
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"

OUTPUT_VIS_DIR = PROJECT_ROOT / "outputs" / "figures"

CSV_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "feature_statistics.csv"

LABELS = ("real", "fake")
MAX_IMAGES_PER_FOLDER = 20

AUGMENTATIONS = {
    "original": DATASET_DIR / "original" / "train",
    "gaussian_blur": DATASET_DIR / "gaussian_blur" / "train",
    "gaussian_noise": DATASET_DIR / "gaussian_noise" / "train",
    "jpeg_compression": DATASET_DIR / "jpeg_compression" / "train",
}


def get_detector_configs():
    """Create feature detectors for all supported methods."""
    return {
        "SIFT": cv2.SIFT_create(),
        "ORB": cv2.ORB_create(nfeatures=1000),
        "AKAZE": cv2.AKAZE_create(),
    }


def get_image_paths(folder: Path):
    """Get all JPG images from folder."""
    image_paths = set()

    for extension in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        image_paths.update(folder.glob(extension))

    return sorted(image_paths)


def get_limited_image_paths(folder: Path, max_images: int | None = MAX_IMAGES_PER_FOLDER):
    """Get image paths with an optional maximum count for demo speed."""
    image_paths = get_image_paths(folder)
    if max_images is None:
        return image_paths
    return image_paths[:max_images]


def load_grayscale_image(image_path: Path):
    """Load image and convert to grayscale."""

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    return image_bgr, image_gray


def detect_features(method_name, detector, image_gray):
    """Detect keypoints and descriptors."""
    keypoints, descriptors = detector.detectAndCompute(image_gray, None)
    return keypoints, descriptors


def compute_descriptor_statistics(descriptors):
    """Compute descriptor statistics."""

    if descriptors is None or len(descriptors) == 0:
        return {
            "descriptor_shape": "None",
            "descriptor_mean": 0,
            "descriptor_variance": 0,
        }

    descriptor_mean = float(np.mean(descriptors))
    descriptor_variance = float(np.var(descriptors))

    return {
        "descriptor_shape": str(descriptors.shape),
        "descriptor_mean": round(descriptor_mean, 4),
        "descriptor_variance": round(descriptor_variance, 4),
    }


def save_keypoint_visualization(
    image_bgr,
    keypoints,
    method_name,
    label,
    augmentation,
    image_name,
):
    """Save keypoint visualization image."""
    output_dir = OUTPUT_VIS_DIR / "feature_extraction" / method_name.lower() / augmentation / label
    output_dir.mkdir(parents=True, exist_ok=True)

    visualized = cv2.drawKeypoints(
        image_bgr,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    output_path = output_dir / f"{image_name}.png"

    cv2.imwrite(str(output_path), visualized)

    return output_path


def count_original_images_by_label():
    """Count original images for each label."""
    counts = {}
    for label in LABELS:
        counts[label] = len(get_image_paths(AUGMENTATIONS["original"] / label))
    return counts


def to_project_relative_path(image_path: Path) -> str:
    """Return a compact project-relative path for CSV output."""
    return image_path.relative_to(PROJECT_ROOT).as_posix()


def process_images():
    """Process real and fake images using SIFT, ORB, and AKAZE."""
    methods = get_detector_configs()

    csv_rows = []

    original_counts = count_original_images_by_label()
    print("\nImage summary from original dataset:")
    print(f"Real images: {original_counts['real']}")
    print(f"Fake images: {original_counts['fake']}")
    print(f"Max images per folder: {MAX_IMAGES_PER_FOLDER}")

    for label in LABELS:
        for augmentation, train_folder in AUGMENTATIONS.items():
            folder = train_folder / label
            all_image_paths = get_image_paths(folder)
            image_paths = get_limited_image_paths(folder)

            print(
                f"\nProcessing label={label} | "
                f"augmentation={augmentation} | "
                f"images={len(image_paths)}/{len(all_image_paths)}"
            )

            for image_index, image_path in enumerate(image_paths):
                image_name = image_path.stem

                image_bgr, image_gray = load_grayscale_image(image_path)

                for method_name, detector in methods.items():
                    keypoints, descriptors = detect_features(
                        method_name,
                        detector,
                        image_gray,
                    )

                    stats = compute_descriptor_statistics(descriptors)

                    save_keypoint_visualization(
                        image_bgr,
                        keypoints,
                        method_name,
                        label,
                        augmentation,
                        image_name,
                    )

                    row = {
                        "image_path": to_project_relative_path(image_path),
                        "label": label,
                        "method": method_name,
                        "augmentation": augmentation,
                        "num_keypoints": len(keypoints),
                        "descriptor_shape": stats["descriptor_shape"],
                        "descriptor_mean": stats["descriptor_mean"],
                        "descriptor_variance": stats["descriptor_variance"],
                    }

                    csv_rows.append(row)

                if (image_index + 1) % 10 == 0:
                    print(f"Processed {image_index + 1}/{len(image_paths)} images")

    return csv_rows


def save_csv(rows):
    """Save feature statistics CSV."""

    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path",
        "label",
        "method",
        "augmentation",
        "num_keypoints",
        "descriptor_shape",
        "descriptor_mean",
        "descriptor_variance",
    ]

    with CSV_OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved CSV statistics to: {CSV_OUTPUT_PATH}")


def main():

    rows = process_images()

    save_csv(rows)

    print("\nFeature extraction completed successfully!")


if __name__ == "__main__":
    main()
