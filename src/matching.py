import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DIR = PROJECT_ROOT / "dataset" / "original" / "train" / "real"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "matching"
SUMMARY_PATH = OUTPUT_DIR / "sift_matching_summary.csv"
RATIO_THRESHOLD = 0.75

AUGMENTATIONS = {
    "gaussian_blur": PROJECT_ROOT / "dataset" / "gaussian_blur" / "train" / "real",
    "gaussian_noise": PROJECT_ROOT / "dataset" / "gaussian_noise" / "train" / "real",
    "jpeg_compression": PROJECT_ROOT
    / "dataset"
    / "jpeg_compression"
    / "train"
    / "real",
}


def find_first_jpg(folder: Path) -> Path:
    """Return the first JPG image in a folder, sorted by filename."""
    jpg_paths = sorted(
        {
            path
            for pattern in ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG")
            for path in folder.glob(pattern)
        }
    )

    if not jpg_paths:
        raise FileNotFoundError(f"No JPG images found in {folder}")

    return jpg_paths[0]


def get_original_image_id(image_path: Path) -> str:
    """Extract the shared image id from an original image filename."""
    suffix = "_original"
    if not image_path.stem.endswith(suffix):
        raise ValueError(f"Expected original image name to end with '{suffix}': {image_path}")

    return image_path.stem[: -len(suffix)]


def find_corresponding_augmented_image(
    original_path: Path,
    augmentation: str,
    augmentation_dir: Path,
) -> Path:
    """Find the augmented image that matches the selected original image."""
    image_id = get_original_image_id(original_path)

    for extension in (".jpg", ".jpeg", ".JPG", ".JPEG"):
        candidate = augmentation_dir / f"{image_id}_{augmentation}{extension}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"No matching {augmentation} image found for {original_path.name} in {augmentation_dir}"
    )


def load_image_pair(image_path: Path) -> tuple:
    """Load an image in color and convert it to grayscale for SIFT."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return image_bgr, image_gray


def detect_sift_features(image_gray, sift):
    """Detect SIFT keypoints and descriptors from a grayscale image."""
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    return keypoints, descriptors


def get_good_matches(descriptors_a, descriptors_b):
    """Match descriptors with BFMatcher and filter them using Lowe's ratio test."""
    if descriptors_a is None or descriptors_b is None or len(descriptors_b) < 2:
        return []

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    knn_matches = matcher.knnMatch(descriptors_a, descriptors_b, k=2)

    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) < 2:
            continue

        m, n = match_pair
        if m.distance < RATIO_THRESHOLD * n.distance:
            good_matches.append(m)

    return good_matches


def save_matches_image(
    image_a,
    keypoints_a,
    image_b,
    keypoints_b,
    good_matches,
    output_path: Path,
) -> None:
    """Draw and save SIFT feature matches between two images."""
    matched_bgr = cv2.drawMatches(
        image_a,
        keypoints_a,
        image_b,
        keypoints_b,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    matched_rgb = cv2.cvtColor(matched_bgr, cv2.COLOR_BGR2RGB)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, matched_rgb)


def write_summary(rows, output_path: Path) -> None:
    """Write SIFT matching counts for each augmentation to CSV."""
    fieldnames = [
        "augmentation",
        "image_a",
        "image_b",
        "keypoints_original",
        "keypoints_augmented",
        "good_matches",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compare_with_augmentation(
    augmentation: str,
    augmentation_dir: Path,
    original_path: Path,
    original_bgr,
    keypoints_original,
    descriptors_original,
    sift,
) -> dict:
    """Run SIFT matching between the original image and one augmentation."""
    augmented_path = find_corresponding_augmented_image(
        original_path,
        augmentation,
        augmentation_dir,
    )
    augmented_bgr, augmented_gray = load_image_pair(augmented_path)
    keypoints_augmented, descriptors_augmented = detect_sift_features(
        augmented_gray,
        sift,
    )

    good_matches = get_good_matches(descriptors_original, descriptors_augmented)
    output_path = OUTPUT_DIR / f"sift_original_vs_{augmentation}.png"
    save_matches_image(
        original_bgr,
        keypoints_original,
        augmented_bgr,
        keypoints_augmented,
        good_matches,
        output_path,
    )

    print(f"\nAugmentation: {augmentation}")
    print(f"Image A path: {original_path}")
    print(f"Image B path: {augmented_path}")
    print(f"Total keypoints original: {len(keypoints_original)}")
    print(f"Total keypoints augmented: {len(keypoints_augmented)}")
    print(f"Total good matches: {len(good_matches)}")
    print(f"Saved matching result to: {output_path}")

    return {
        "augmentation": augmentation,
        "image_a": str(original_path),
        "image_b": str(augmented_path),
        "keypoints_original": len(keypoints_original),
        "keypoints_augmented": len(keypoints_augmented),
        "good_matches": len(good_matches),
    }


def main() -> None:
    original_path = find_first_jpg(ORIGINAL_DIR)
    original_bgr, original_gray = load_image_pair(original_path)

    sift = cv2.SIFT_create()
    keypoints_original, descriptors_original = detect_sift_features(original_gray, sift)

    summary_rows = [
        compare_with_augmentation(
            augmentation,
            augmentation_dir,
            original_path,
            original_bgr,
            keypoints_original,
            descriptors_original,
            sift,
        )
        for augmentation, augmentation_dir in AUGMENTATIONS.items()
    ]

    write_summary(summary_rows, SUMMARY_PATH)
    print(f"\nSaved CSV summary to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
