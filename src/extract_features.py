"""
extract_features.py
Extract features for all images referenced in a labels CSV and save to CSV.
"""

import argparse
import os
from src.data_utils import read_labels, build_feature_dataframe
from src.utils import ensure_dir

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", required=True, help="CSV with filename,label")
    p.add_argument("--out", required=True, help="Output features CSV path")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(os.path.dirname(args.out) or ".")
    labels_df = read_labels(args.data_csv)
    feats_df = build_feature_dataframe(labels_df)
    cols = ['filename','crack_count','crack_length','shear_length','label']
    feats_df = feats_df[cols]
    feats_df.to_csv(args.out, index=False)
    print(f"Saved features to {args.out} (rows: {len(feats_df)})")

if __name__ == "__main__":
    main()
