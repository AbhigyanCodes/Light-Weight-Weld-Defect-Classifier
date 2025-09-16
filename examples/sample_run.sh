#!/usr/bin/env bash
set -e
# This script demonstrates the full pipeline on sample data (images in data/samples/)
python src/extract_features.py --data-csv data/labels.csv --out features/features.csv
python src/train.py --data-csv data/labels.csv --output-dir models --epochs 10 --batch-size 8 --save-features
python src/predict_batch.py --model models/welding_classifier.h5 --features features/features.csv --out predictions/preds.csv
python src/visualize.py --features features/features.csv --cv-metrics models/cv_metrics.csv --predictions predictions/preds.csv --out-dir docs/figures
