import csv
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"

OUTPUT_VIS_DIR = PROJECT_ROOT / "outputs" / "figures"

CSV_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "feature_statistics.csv"


AUGMENTATIONS = {
    "original": DATASET_DIR / "original" / "train" / "real",
    "gaussian_blur": DATASET_DIR / "gaussian_blur" / "train" / "real",
    "gaussian_noise": DATASET_DIR / "gaussian_noise" / "train" / "real",
    "jpeg_compression": DATASET_DIR / "jpeg_compression" / "train" / "real",
}


def get_image_paths(folder: Path):
    """Get all JPG images from folder."""

    image_paths = []

    for extension in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
        image_paths.extend(folder.glob(extension))

    return sorted(image_paths)


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
    augmentation,
    image_name,
):
    """Save keypoint visualization image."""

    output_dir = OUTPUT_VIS_DIR / method_name.lower()
    output_dir.mkdir(parents=True, exist_ok=True)

    visualized = cv2.drawKeypoints(
        image_bgr,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )

    output_path = output_dir / f"{image_name}_{augmentation}.png"

    cv2.imwrite(str(output_path), visualized)

    print(f"Saved visualization: {output_path}")


def process_images():
    """Process all images using ORB and AKAZE."""

    orb = cv2.ORB_create(
        nfeatures=1000
    )

    akaze = cv2.AKAZE_create()

    methods = {
        "ORB": orb,
        "AKAZE": akaze,
    }

    csv_rows = []

    for augmentation, folder in AUGMENTATIONS.items():

        image_paths = get_image_paths(folder)

        print(f"\nProcessing augmentation: {augmentation}")
        print(f"Found {len(image_paths)} images")

        for image_path in image_paths:

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
                    augmentation,
                    image_name,
                )

                row = {
                    "image_path": str(image_path),
                    "method": method_name,
                    "augmentation": augmentation,
                    "num_keypoints": len(keypoints),
                    "descriptor_shape": stats["descriptor_shape"],
                    "descriptor_mean": stats["descriptor_mean"],
                    "descriptor_variance": stats["descriptor_variance"],
                }

                csv_rows.append(row)

                print(
                    f"{method_name} | "
                    f"{augmentation} | "
                    f"{image_name} | "
                    f"Keypoints: {len(keypoints)}"
                )

    return csv_rows


def save_csv(rows):
    """Save feature statistics CSV."""

    CSV_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "image_path",
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