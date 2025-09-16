#!/usr/bin/env bash
set -e
python src/predict_batch.py --model models/welding_classifier.h5 --features features/features.csv --out predictions/preds.csv
