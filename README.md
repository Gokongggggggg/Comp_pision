# Deepfake Detection using Classical Computer Vision Feature Descriptors

## Project Overview

This project studies whether classical computer vision feature descriptors can
support deepfake detection and robustness analysis. The dataset is based on
**Celeb-DF v2**, with two classes:

- `real`
- `fake`

The project focuses on three local feature descriptors:

- SIFT
- ORB
- AKAZE

The main idea is to observe how these descriptors behave under common image
augmentations and whether extracted statistics can be used for classical
machine learning classification. The project includes descriptor robustness
analysis, feature matching visualization, and real vs fake classification using
statistical feature vectors.

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook
```

Run the notebooks in this order:

1. `notebooks/01_preprocess.ipynb`
2. `notebooks/02_augmentation.ipynb`
3. `notebooks/03_feature_extraction.ipynb`
4. `notebooks/04_matching_analysis.ipynb`
5. `notebooks/05_machine_learning.ipynb`

The raw Celeb-DF v2 folders are not included in this repository. Download the
dataset from the official Celeb-DF v2 source and place the required folders in
the project root before running preprocessing.

## Pipeline Overview

### 1. Preprocessing

The preprocessing phase prepares face crops from video frames.

- Extract frames from video
- Detect faces
- Align faces using facial landmarks
- Crop aligned faces to `112x112`
- Save processed face images and metadata

### 2. Augmentation

The augmentation phase creates controlled image distortions for robustness
analysis.

- Original image
- Gaussian blur
- Gaussian noise
- JPEG compression

The output is stored in:

```text
dataset/{augmentation}/{split}/{class}/*.jpg
```

### 3. Feature Extraction

The feature extraction phase computes local descriptors from real and fake
images across all augmentation types.

- SIFT keypoint detection and descriptor extraction
- ORB keypoint detection and descriptor extraction
- AKAZE keypoint detection and descriptor extraction
- Descriptor statistics: mean and variance
- Keypoint statistics: number of detected keypoints

Main script:

```bash
python src/feature_extraction.py
```

Main output:

```text
outputs/feature_statistics.csv
outputs/figures/feature_extraction/
```

### 4. Feature Matching

The feature matching phase compares original images with augmented images.

- BFMatcher
- `NORM_L2` for SIFT
- `NORM_HAMMING` for ORB and AKAZE
- Lowe Ratio Test
- Matching visualization for real and fake samples

Main script:

```bash
python src/matching.py
```

Main output:

```text
outputs/matching/all_matching_summary.csv
outputs/matching/real/
outputs/matching/fake/
```

### 5. Machine Learning

The machine learning phase uses descriptor and matching statistics as tabular
features for real vs fake classification.

- Random Forest
- Support Vector Machine (SVM)
- Accuracy, precision, recall, and F1-score
- Classification report
- Confusion matrix visualization

Main script:

```bash
python src/ml_pipeline.py
```

Main output:

```text
outputs/results/ml_evaluation_results.csv
outputs/figures/confusion_matrix_random_forest.png
outputs/figures/confusion_matrix_svm.png
```

## Repository Structure

```text
.
├── notebooks/
│   ├── 01_preprocess.ipynb
│   ├── 02_augmentation.ipynb
│   ├── 03_feature_extraction.ipynb
│   ├── 04_matching_analysis.ipynb
│   └── 05_machine_learning.ipynb
├── src/
│   ├── feature_extraction.py
│   ├── matching.py
│   ├── ml_pipeline.py
│   └── visualize_matching.py
├── outputs/
│   ├── figures/
│   ├── matching/
│   └── results/
├── dataset/
│   ├── original/
│   ├── gaussian_blur/
│   ├── gaussian_noise/
│   └── jpeg_compression/
├── dataset_manifest.csv
├── requirements.txt
└── README.md
```

## Notebook Description

### `01_preprocess.ipynb`

Prepares the Celeb-DF v2 data for image-based analysis. The notebook extracts
frames, detects faces, aligns them using facial landmarks, and saves `112x112`
face crops.

### `02_augmentation.ipynb`

Generates augmented image versions from the preprocessed face crops. The
augmentations are Gaussian blur, Gaussian noise, and JPEG compression, plus the
original image as the control condition.

### `03_feature_extraction.ipynb`

Demonstrates feature extraction using SIFT, ORB, and AKAZE. It explains
keypoints, descriptors, descriptor shape, descriptor statistics, and keypoint
visualization.

### `04_matching_analysis.ipynb`

Demonstrates feature matching between original and augmented images. It compares
SIFT, ORB, and AKAZE using BFMatcher and Lowe Ratio Test, then visualizes good
matches.

### `05_machine_learning.ipynb`

Demonstrates the classical machine learning phase. It loads feature statistics,
prepares labels, trains Random Forest and SVM models, evaluates them, and
visualizes confusion matrices.

## Dataset and Manifest

This project uses **Celeb-DF v2**, a deepfake dataset containing real celebrity
videos and synthesized fake videos.

The generated `dataset_manifest.csv` is the source of truth for downstream
experiments. It records the image path, augmentation type, split, and class
label.

Important columns:

| Column | Description |
|---|---|
| `original_path` | Historical path from the preprocessing phase |
| `augmented_path` | Image path relative to the project root |
| `augmentation` | `original`, `gaussian_blur`, `gaussian_noise`, or `jpeg_compression` |
| `label` | Numeric label, where `0 = real` and `1 = fake` |
| `split` | Dataset split, such as `train` or `test` |
| `class` | String label, either `real` or `fake` |

Example usage:

```python
import pandas as pd
import cv2

df = pd.read_csv("dataset_manifest.csv")
train_real = df[(df["split"] == "train") & (df["class"] == "real")]

row = train_real.iloc[0]
image = cv2.imread(row["augmented_path"])
label = row["label"]
```

## Experimental Results

The experiments indicate several useful observations:

- SIFT generally produced more stable matches than ORB and AKAZE in the sample
  matching analysis.
- JPEG compression preserved many local structures compared with stronger
  random distortions.
- Gaussian noise changed local descriptor patterns and affected matching
  robustness.
- Random Forest outperformed SVM in the current machine learning experiment.
- Augmentation changes descriptor robustness, keypoint counts, and matching
  quality.

These results are useful for understanding how classical feature descriptors
respond to image degradation. They should be interpreted as project-level
experimental findings rather than universal conclusions.

## Technologies Used

- Python
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook

## Team Collaboration

This project is organized for collaborative development and presentation:

- GitHub is used for version control and collaboration.
- Core processing logic is placed in modular scripts under `src/`.
- Jupyter notebooks are used for step-by-step demonstrations and final project
  presentation.
- Generated outputs are organized under `outputs/` for figures, matching
  results, and machine learning evaluation results.

## Notes

Large generated data folders such as `dataset/`, processed faces, and output
artifacts may be excluded from Git depending on `.gitignore` settings. If a file
is missing, rerun the corresponding notebook or script phase.
