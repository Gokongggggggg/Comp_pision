# Deepfake Detection — Preprocessing & Augmentation Pipeline

Pipeline preprocessing + augmentation untuk dataset **Celeb-DF v2** sebagai input
deepfake detection. Output: face crops 112×112 yang sudah di-align + 4 versi
augmented per crop, lengkap dengan manifest CSV.

---

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook
```

Buka notebook secara berurutan: `notebooks/01_preprocess.ipynb` → `notebooks/02_augment.ipynb`.

Data mentah (`Celeb-real/`, `Celeb-synthesis/`) tidak ada di repo (gitignore).
Download dari [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics)
dan letakkan di root project sebelum run `01_preprocess.ipynb`.

---

## 01_preprocess.ipynb — Phase 1

**Tujuan**: extract & align face crops dari video Celeb-DF v2.

**Pipeline**:
1. Parse `List_of_testing_videos.txt` (split resmi Celeb-DF; tidak disertakan di
   repo, download dari [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics))
   → tentukan train/test split resmi (anti data leakage berdasarkan identitas).
2. Subsample fake-train supaya seimbang dengan real-train (≈ 4.8k vs 4.8k).
3. Sample frame *evenly spaced* per video (10 frame default).
4. Deteksi wajah pakai **MTCNN** (`facenet-pytorch`) — dapat bounding box + 5
   landmark.
5. Filter confidence ≥ 0.90 → buang false positive.
6. **Similarity transform** ke template ArcFace 112×112 (5-point alignment).
7. Simpan JPG (q=95) + tulis `faces_manifest.csv`.

**Output** (one-time, tidak ada di repo): `processed_faces/{train,test}/{real,fake}/*.jpg`
(13,999 crops total) + `processed_faces/faces_manifest.csv`. Notebook ini
dijalankan sekali untuk generate input Phase 2; setelah `dataset/` terbentuk,
folder ini boleh dihapus.

---

## 02_augment.ipynb — Phase 2

**Tujuan**: produce 4 versi per face crop untuk evaluasi robustness descriptor.

**4 kondisi**:

| Augmentation | Parameter | Simulates |
|---|---|---|
| `original` | — (copy) | Control group |
| `jpeg_compression` | q=30 roundtrip | Social media re-compression |
| `gaussian_blur` | 5×5 kernel | Low-quality camera blur |
| `gaussian_noise` | σ=25 | Sensor noise / low-light |

**Input**: `processed_faces/faces_manifest.csv` + crops Phase 1.
**Output**: `dataset/{aug}/{split}/{class}/{stem}_{aug}.jpg` (55,996 file total)
+ `dataset_manifest.csv` di root.

Seed `np.random.seed(42)` untuk reproducibility noise.

---

## dataset_manifest.csv — cara pakai

File CSV ini adalah **source of truth** untuk downstream task (descriptor
extraction, classifier training, evaluation). Berisi 55,996 baris (13,999 crop
× 4 aug).

**Kolom**:

| Kolom | Contoh | Keterangan |
|---|---|---|
| `original_path` | `train/real/id59_0008_f0000.jpg` | Path historis dari Phase 1 (relative ke `processed_faces/` saat di-generate). Untuk load gambar pakai `augmented_path` saja. |
| `augmented_path` | `dataset/original/train/real/id59_0008_f0000_original.jpg` | Relative ke project root |
| `augmentation` | `original` / `jpeg_compression` / `gaussian_blur` / `gaussian_noise` | Jenis augmentasi |
| `label` | `0` / `1` | 0 = real, 1 = fake |
| `split` | `train` / `test` | Sesuai Celeb-DF official split |
| `class` | `real` / `fake` | String label (redundan dengan `label`, buat convenience) |

**Contoh penggunaan**:

```python
import pandas as pd
import cv2

df = pd.read_csv("dataset_manifest.csv")

# Train set, augmentasi blur saja
train_blur = df[(df["split"] == "train") & (df["augmentation"] == "gaussian_blur")]

# Load satu gambar
row = train_blur.iloc[0]
img = cv2.imread(row["augmented_path"])  # path sudah relative ke project root
y = row["label"]                          # 0 / 1
```

**Pola filtering yang umum**:

```python
# Per-augmentation comparison
for aug in df["augmentation"].unique():
    subset = df[df["augmentation"] == aug]
    # ... extract descriptor, evaluate, etc.

# Per-class breakdown
df.groupby(["split", "class", "augmentation"]).size()
```

---

## Project layout

```
.
├── notebooks/
│   ├── 01_preprocess.ipynb    # Phase 1 — video → aligned face crops
│   └── 02_augment.ipynb       # Phase 2 — face crops → 4 augmented versions
├── README.md
├── requirements.txt
├── .gitignore
├── dataset_manifest.csv       # Manifest 55,996 augmented crops (gitignored)
└── dataset/                   # Phase 2 output (gitignored)
    ├── original/{train,test}/{real,fake}/*.jpg
    ├── jpeg_compression/...
    ├── gaussian_blur/...
    └── gaussian_noise/...
```

---

## Dataset

**Celeb-DF v2** — 590 video real + 5,639 video fake selebriti.

- Train: 4,766 real + 4,789 fake aligned crops
- Test: 1,069 real + 3,375 fake aligned crops
- `YouTube-real/` tidak dipakai (bukan selebriti, tidak match dengan synthesis)

Split mengikuti `List_of_testing_videos.txt` resmi dari paper Celeb-DF — anti
data leakage berdasarkan identitas, bukan random split. Info split sudah baked
ke dalam `dataset_manifest.csv` (kolom `split`), file txt asli tidak disertakan
di repo.
