"""
evaluate.py
Evaluate saved model against a feature CSV or the original labels CSV.
Prints classification report and saves confusion matrix CSV if requested.
"""

import argparse
import os
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from src.data_utils import read_labels, build_feature_dataframe

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--data-csv", required=True)
    p.add_argument("--features-csv", required=False)
    p.add_argument("--out-cm", required=False, help="Path to save confusion matrix CSV")
    return p.parse_args()

def main():
    args = parse_args()
    labels_df = read_labels(args.data_csv)
    if args.features_csv:
        feats_df = pd.read_csv(args.features_csv)
    else:
        feats_df = build_feature_dataframe(labels_df)

    feature_cols = ['crack_count','crack_length','shear_length']
    scaler = joblib.load(os.path.join(os.path.dirname(args.model), "scaler.save"))
    X = scaler.transform(feats_df[feature_cols].values)
    y = feats_df['label'].values.astype(int)

    model = load_model(args.model)
    preds = (model.predict(X) > 0.5).astype(int).reshape(-1)

    print("Classification report:")
    print(classification_report(y, preds, zero_division=0))
    cm = confusion_matrix(y, preds)
    print("Confusion matrix:")
    print(cm)
    if args.out_cm:
        os.makedirs(os.path.dirname(args.out_cm) or ".", exist_ok=True)
        pd.DataFrame(cm).to_csv(args.out_cm, index=False)
        print(f"Saved confusion matrix to {args.out_cm}")

if __name__ == "__main__":
    main()
