"""
predict.py
Predict single image using saved model and scaler.
"""

import argparse
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from src.data_utils import extract_features

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to model .h5")
    p.add_argument("--image", required=True, help="Path to image")
    return p.parse_args()

def main():
    args = parse_args()
    model_dir = os.path.dirname(args.model)
    scaler_path = os.path.join(model_dir, "scaler.save")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("scaler.save not found in model directory")
    scaler = joblib.load(scaler_path)
    feats = extract_features(args.image)
    feature_vector = np.array([[feats['crack_count'], feats['crack_length'], feats['shear_length']]])
    X = scaler.transform(feature_vector)
    model = load_model(args.model)
    prob = float(model.predict(X)[0][0])
    label = int(prob > 0.5)
    print(f"Image: {args.image}")
    print(f"Predicted probability (welded): {prob:.4f}")
    print(f"Predicted label: {label} (1=welded, 0=non-welded)")

if __name__ == "__main__":
    main()
