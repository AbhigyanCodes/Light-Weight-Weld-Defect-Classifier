# Light-Weight-Weld-Defect-Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)]
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)]
[![Build](https://img.shields.io/badge/build-manual-orange.svg)]

> Lightweight Machine Learning for Real-Time Defect Detection and Categorization in Industrial Welds.  
> Hybrid approach using fast numerical feature extraction (crack_count, crack_length, shear_length) and a compact ML model (MLP). Optional CNN+Dense fusion model is included for extension.

---

## Table of contents
- [Overview](#overview)  
- [Features](#features)  
- [Repository structure](#repository-structure)  
- [Dataset & Hardware (optional)](#dataset--hardware-optional)  
- [Software & dependencies](#software--dependencies)  
- [Quick start (minimal)](#quick-start-minimal)  
- [Running the pipeline](#running-the-pipeline)  
- [Configuration](#configuration)  
- [Testing & safety checklist](#testing--safety-checklist)  
- [Logging & debugging](#logging--debugging)  
- [Troubleshooting](#troubleshooting)  
- [Contributing](#contributing)  
- [License](#license)  
- [Acknowledgements](#acknowledgements)

---

## Overview
This repository contains the code, scripts, and documentation to reproduce experiments from the paper:

**"Lightweight Machine Learning for Real-Time Defect Detection and Categorization in Industrial Welds"**.

The pipeline extracts fast numerical features from weld images (crack count, crack length, shear length) using OpenCV/Skimage heuristics, trains a lightweight MLP classifier on these features, and provides tools for batch prediction, evaluation, and visualization. A CNN+Dense fusion model constructor is included for future image+numerical fusion experiments.

The project emphasizes reproducibility and edge deployment readiness (small model size, fast inference).

---

## Features
- Fast numerical feature extraction: `crack_count`, `crack_length`, `shear_length`.  
- Lightweight MLP classifier trained with Stratified K-Fold cross-validation.  
- Optional CNN + Dense fusion model constructor (for future extensions).  
- Scripts for: feature extraction, training, batch prediction, evaluation, and visualization.  
- Utilities to save scaler and best model for inference.  
- Example notebook and sample images for quick demo and CI smoke tests.

---

## Repository structure
```
Light-Weight-Weld-Defect-Classifier/
├── .github/
│   └── workflows/ci.yml
├── data/
│   ├── images/
│   └── labels.csv
├── datasets/
│   └── get_data.sh
├── docs/
│   ├── figures/
│   └── paper/062-NCWAM2025.pdf
├── examples/
│   └── sample_run.sh
├── models/
├── notebooks/
│   └── Weld_Defect_Classifier_cleaned.ipynb
├── src/
│   ├── data_utils.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   ├── extract_features.py
│   ├── predict.py
│   ├── predict_batch.py
│   ├── evaluate.py
│   └── visualize.py
├── scripts/
│   └── run_*.sh
├── tests/
├── requirements.txt
├── Dockerfile
├── LICENSE
└── README.md
```

---

## Dataset & Hardware (optional)
The pipeline works with images of welded samples. For best results:
- High-resolution images of weld region (consistent lighting and camera-view).  
- A sample staging rig or fixed mount to keep images aligned helps reduce noise.  
- Optional: ring-light or diffuse illumination to reduce shadows.

Dataset CSV format (`data/labels.csv`):
```
filename,label
data/images/img_0001.jpg,1
data/images/img_0002.jpg,0
```
- `label`: `1` => welded, `0` => non-welded

Place sample images in `data/images/`. Large datasets should be stored outside git; use `datasets/get_data.sh` to download.

---

## Software & dependencies
- Python 3.10+  
- Key libraries (see `requirements.txt`):
  - `numpy`, `pandas`, `scikit-learn`, `opencv-python`, `scikit-image`, `matplotlib`, `seaborn`, `tensorflow`, `joblib`, `tqdm`
- (Optional) `conda` environment via `environment.yml`
- Dockerfile included for reproducible container runs.

Install quickly:
```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

---

## Quick start (minimal)
1. Clone:
```bash
git clone https://github.com/AbhigyanCodes/Light-Weight-Weld-Defect-Classifier.git
cd Light-Weight-Weld-Defect-Classifier
```

2. Prepare dataset:
- Put images in `data/images/` and create `data/labels.csv` as above.

3. Extract features:
```bash
python src/extract_features.py --data-csv data/labels.csv --out features/features.csv
```

4. Train model (Stratified K-Fold, saves best model & scaler):
```bash
python src/train.py --data-csv data/labels.csv --output-dir models --epochs 50 --batch-size 16 --save-features
```

5. Batch predict:
```bash
python src/predict_batch.py --model models/welding_classifier.h5 --features features/features.csv --out predictions/preds.csv
```

6. Evaluate:
```bash
python src/evaluate.py --model models/welding_classifier.h5 --data-csv data/labels.csv --features-csv features/features.csv
```

7. Visualize:
```bash
python src/visualize.py --features features/features.csv --cv-metrics models/cv_metrics.csv --predictions predictions/preds.csv --out-dir docs/figures
```

---

## Running the pipeline
- Use the shell helpers in `scripts/` for convenience:
  - `scripts/run_extract_features.sh`
  - `scripts/run_train.sh`
  - `scripts/run_predict_batch.sh`
  - `scripts/run_visualize.sh`

- A ready example `examples/sample_run.sh` runs the entire pipeline on the sample dataset (if present).

---

## Configuration
Key configuration is available via CLI args and small config sections in scripts:

- `src/train.py` arguments:
  - `--data-csv` : path to labels CSV
  - `--features-csv` : optional precomputed features CSV
  - `--output-dir` : models output directory
  - `--epochs`, `--batch-size`, `--n-splits`

- `src/visualize.py`:
  - `--features`, `--cv-metrics`, `--predictions`, `--out-dir`

Advanced settings:
- Edit model constructors in `src/model.py` to change architectures (MLP vs fusion CNN+Dense).
- Adjust `src/data_utils.py` feature extraction thresholds for different lighting/quality.

---

## Testing & safety checklist
- [ ] Ensure images are anonymized / carry no restricted info before sharing.  
- [ ] Test feature extraction on a small sample set: `python -c "from src.data_utils import extract_features; print(extract_features('data/images/sample.jpg'))"`  
- [ ] Run unit tests:
```bash
pip install pytest
pytest -q
```
- [ ] Validate model predictions on a held-out test set before deployment.
- [ ] When deploying to edge hardware, verify model size and inference speed (quantize/prune if needed).

---

## Logging & debugging
- Scripts print progress to stdout. Use `--verbose` flags if added to scripts.
- Saved artifacts (models, scaler, metrics) are in `models/`.
- For service deployments, run within systemd or Docker and forward logs to stdout or a log file.

---

## Troubleshooting
**Common issues**
- `FileNotFoundError` for images: ensure `data/labels.csv` filenames are correct and relative to repo root.  
- Poor features due to lighting: add diffuse lighting or normalize images; consider histogram equalization.  
- TensorFlow GPU errors: ensure correct CUDA/cuDNN versions (or use CPU-only).  
- Model not improving: try data augmentation, increase epochs, or use CNN+fusion model.

If stuck, open an issue with: dataset sample, exact command run, and stack trace.

---

## Contributing
Contributions welcome! Workflow:
1. Fork the repo.  
2. Create a feature branch: `git checkout -b feature/your-feature`  
3. Add tests and docs for your changes.  
4. Open a PR with a clear description and testing steps.

See `CONTRIBUTING.md` for details.

---

## License
This project is released under the **MIT License**. See `LICENSE` for full details.

---

## Acknowledgements
- Project authors: Kavita Jaiswal, Shashi Tiwari, Abhigyan Patnaik, Mohd Kaif, Juttuka Saaketh, Archana Sharma.  
- Libraries and tools: OpenCV, scikit-image, scikit-learn, TensorFlow, seaborn, matplotlib.  
- Paper: "Lightweight Machine Learning for Real-Time Defect Detection and Categorization in Industrial Welds" (see `docs/paper/062-NCWAM2025.pdf`).

---

