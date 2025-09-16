#!/usr/bin/env bash
set -e
python src/extract_features.py --data-csv data/labels.csv --out features/features.csv
