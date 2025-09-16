"""
predict_batch.py
Load features CSV and model, generate predictions CSV with prob & label.
"""

import argparse
import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    feats_df = pd.read_csv(args.features)
    feature_cols = ['crack_count','crack_length','shear_length']
    if not all(c in feats_df.columns for c in feature_cols):
        raise ValueError(f"Features CSV must contain {feature_cols}")

    model_dir = os.path.dirname(args.model)
    scaler = joblib.load(os.path.join(model_dir, "scaler.save"))
    X = scaler.transform(feats_df[feature_cols].values)
    model = load_model(args.model)
    probs = model.predict(X).reshape(-1)
    preds = (probs > 0.5).astype(int)
    out_df = feats_df.copy()
    out_df['pred_prob'] = probs
    out_df['pred_label'] = preds
    out_df.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out} (rows: {len(out_df)})")

if __name__ == "__main__":
    main()
