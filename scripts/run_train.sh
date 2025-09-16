#!/usr/bin/env bash
set -e
python src/train.py --data-csv data/labels.csv --output-dir models --epochs 50 --batch-size 16
