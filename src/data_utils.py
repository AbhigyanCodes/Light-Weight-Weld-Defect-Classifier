"""
data_utils.py
Feature extraction, loading labels, basic preprocessing utilities.
"""

import os
from typing import Dict, List
import cv2
import numpy as np
import pandas as pd
from skimage import measure
from sklearn.preprocessing import StandardScaler

def read_labels(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if 'filename' not in df.columns or 'label' not in df.columns:
        raise ValueError("labels CSV must contain 'filename' and 'label' columns")
    return df

def basic_preprocess(img: np.ndarray, size=(256, 256)) -> np.ndarray:
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if img_resized.ndim == 3:
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_resized
    return gray

def extract_features(image_path: str) -> Dict[str, float]:
    """
    Heuristic features:
      - crack_count : number of connected components in edge image (area filter)
      - crack_length: sum of perimeters of those components
      - shear_length: longest near-horizontal line from HoughLinesP
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    gray = basic_preprocess(img)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Morphological cleanup
    kernel = np.ones((3,3), np.uint8)
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    labels = measure.label(edges_closed, connectivity=2)
    props = measure.regionprops(labels)

    crack_count = 0
    crack_length = 0.0
    for p in props:
        if p.area < 20:
            continue
        crack_count += 1
        crack_length += p.perimeter

    # Shear length via Hough
    lines = cv2.HoughLinesP(edges_closed, rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    shear_length = 0.0
    if lines is not None:
        lengths = []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            length = np.hypot(x2-x1, y2-y1)
            dx = x2-x1
            dy = y2-y1
            if abs(dy) < 0.4 * max(1, abs(dx)):
                lengths.append(length)
        if lengths:
            shear_length = float(max(lengths))
    return {
        "crack_count": float(crack_count),
        "crack_length": float(crack_length),
        "shear_length": float(shear_length)
    }

def build_feature_dataframe(labels_df: pd.DataFrame, features_cache: dict = None) -> pd.DataFrame:
    rows = []
    for _, r in labels_df.iterrows():
        fname = r['filename']
        label = int(r['label'])
        if features_cache and fname in features_cache:
            feats = features_cache[fname]
        else:
            feats = extract_features(fname)
        feats['filename'] = fname
        feats['label'] = label
        rows.append(feats)
    return pd.DataFrame(rows)

def standardize_features(df: pd.DataFrame, feature_cols=None):
    if feature_cols is None:
        feature_cols = ['crack_count', 'crack_length', 'shear_length']
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    X_df = pd.DataFrame(X, columns=feature_cols, index=df.index)
    return X_df, scaler
