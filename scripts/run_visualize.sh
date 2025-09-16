#!/usr/bin/env bash
set -e
python src/visualize.py --features features/features.csv --cv-metrics models/cv_metrics.csv --predictions predictions/preds.csv --out-dir docs/figures
