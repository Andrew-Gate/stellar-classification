# 🌌 Stellar Classification

A Data Mining project for classifying celestial objects — **Stars**, **Galaxies**, and **Quasars (QSO)** — using machine learning techniques applied to the Sloan Digital Sky Survey (SDSS) dataset.

---

## 📌 Project Overview

The dataset contains **100,000 observations** from the SDSS catalog, each described by 18 photometric and spectroscopic features (e.g., photometric filters u, g, r, i, z, redshift, coordinates). The goal is to train and compare multiple classifiers on a 3-class target: `GALAXY` (59.4%), `STAR` (21.6%), `QSO` (19.0%).

The full pipeline covers:
- Exploratory Data Analysis (EDA)
- Data preprocessing (stratified sampling, cleaning, scaling, SMOTE balancing)
- Training and evaluation of base and ensemble classifiers


---

## 📊 Dataset

- **Source**: [SDSS Star Classification Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17)
- **Size**: 100,000 instances × 18 features
- **Target classes**: `GALAXY` · `STAR` · `QSO`
- **No missing values**; anomalous values in photometric filters (e.g., `-9999`) handled during preprocessing

### Feature Description

| Feature | Description |
|---------|-------------|
| `alpha` | Right ascension angle (celestial coordinate) |
| `delta` | Declination angle (celestial coordinate) |
| `u` | Ultraviolet photometric filter magnitude |
| `g` | Green photometric filter magnitude |
| `r` | Red photometric filter magnitude |
| `i` | Near-infrared photometric filter magnitude |
| `z` | Infrared photometric filter magnitude |
| `redshift` | Spectroscopic redshift value |
| `run_ID`, `cam_col`, `field_ID` | SDSS imaging run identifiers |
| `plate`, `MJD`, `fiber_ID` | SDSS spectroscopic observation identifiers |

---

## ⚙️ Pipeline

### 1. Exploratory Data Analysis (`notebooks/01_esplorazione_stellare.ipynb`)
- Dataset shape, types, and null value inspection
- Class distribution analysis — dataset is imbalanced (GALAXY ~59%)
- Descriptive statistics and outlier detection (e.g., `-9999` sentinel values in `u`, `g`, `z`)
- Histograms, boxplots, and correlation heatmap of numerical features

### 2. Preprocessing — Phase 1 (`02_preprocessing.py`)
- **Stratified sampling**: 15,000 instances extracted from 100k preserving class ratios
- **Cleaning**: duplicate removal and null value check
- **Target encoding**: `GALAXY → 0`, `STAR → 1`, `QSO → 2`
- Output: `data_sample_15k.csv`

### 3. Preprocessing — Phase 2 (`03_preprocessing_final.py`)
- **Feature scaling**: StandardScaler (Z-score normalization) applied to all numerical features
- **Class balancing**: SMOTE (Synthetic Minority Oversampling Technique) to equalize class sizes
- Output: `data_ready.csv`

### 4. Base Models (`04_models_base.py`)
Trained with a **70/30 stratified train-test split**:

| Model | Notes |
|-------|-------|
| 🌳 Decision Tree | Interpretable, fast, prone to overfitting |
| 🔵 K-Nearest Neighbors | k=5, distance-based |
| ⚙️ Support Vector Machine | RBF kernel |
| 🧠 Naive Bayes | Gaussian, probabilistic |
| 🧠 Neural Network (MLP) | 1 hidden layer, 50 neurons, max_iter=500 |

### 5. Ensemble Models (`05_models_ensemble.py`)
Trained with an **80/20 stratified train-test split**:

| Model | Notes |
|-------|-------|
| 🌲 Random Forest | 100 estimators |
| 📦 Bagging | Decision Tree base, 100 estimators |
| 🚀 AdaBoost | Decision Tree stump, 100 estimators, lr=1.0 |

### 6. Evaluation (`06_evaluation.py`)
- Unified comparison of all **8 models** on **Accuracy** and **F1-macro**
- Results sorted by accuracy and visualized as a horizontal bar chart
- Scaling applied selectively (KNN, SVM, MLP use StandardScaler; tree-based models do not)

---

## 🛠️ Requirements

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib
```

> Python **3.8+** recommended.

---

## 🚀 How to Run

```bash
# Step 1 – Exploratory Data Analysis
jupyter notebook notebooks/01_esplorazione_stellare.ipynb

# Step 2 – Preprocessing Phase 1 (sampling + encoding)
python 02_preprocessing.py

# Step 3 – Preprocessing Phase 2 (scaling + SMOTE)
python 03_preprocessing_final.py

# Step 4 – Train base models
python 04_models_base.py

# Step 5 – Train ensemble models
python 05_models_ensemble.py

# Step 6 – Evaluate and compare all models
python 06_evaluation.py
```

---

## 📁 Notes on Data Files

The raw dataset (`star_classification.csv`, ~100k rows) and processed files (`data_ready.csv`, `data_sample_15k.csv`) are **not included** in the repository due to size constraints.  
You can download the original dataset from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17) and place it in the `data/` folder before running the pipeline.

---

## 📚 References

- SDSS DR17: https://www.sdss.org/dr17/
- Kaggle Dataset: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
- scikit-learn Documentation: https://scikit-learn.org/
- imbalanced-learn (SMOTE): https://imbalanced-learn.org/

---

*Data Mining Project — University of Calabria*
