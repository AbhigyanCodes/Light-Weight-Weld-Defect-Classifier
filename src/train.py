"""
train.py
Training script for the MLP (default). Supports Stratified K-Fold, saving best model, scaler, and cv metrics.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
import joblib
from src.data_utils import read_labels, build_feature_dataframe, standardize_features
from src.model import create_mlp
from src.utils import ensure_dir, setup_logger, save_json

logger = setup_logger("train")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-csv", required=True, help="CSV with filename,label")
    p.add_argument("--features-csv", required=False, help="If you have features precomputed")
    p.add_argument("--output-dir", default="models")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--save-features", action="store_true", help="Save computed features to models/features_saved.csv")
    return p.parse_args()

def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    labels_df = read_labels(args.data_csv)
    if args.features_csv:
        feats_df = pd.read_csv(args.features_csv)
        if 'label' not in feats_df.columns:
            mapping = dict(zip(labels_df['filename'], labels_df['label']))
            feats_df['label'] = feats_df['filename'].map(mapping)
    else:
        logger.info("Extracting features for all samples (this may take time)...")
        feats_df = build_feature_dataframe(labels_df)

    if args.save_features:
        outp = os.path.join(args.output_dir, "features_saved.csv")
        feats_df.to_csv(outp, index=False)
        logger.info(f"Saved features to {outp}")

    feature_cols = ['crack_count', 'crack_length', 'shear_length']
    X_df, scaler = standardize_features(feats_df, feature_cols)
    X = X_df.values
    y = feats_df['label'].values.astype(int)

    # save scaler
    joblib.dump(scaler, os.path.join(args.output_dir, "scaler.save"))

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    best_f1 = -1.0
    best_model_path = os.path.join(args.output_dir, "welding_classifier.h5")
    metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        logger.info(f"Starting fold {fold}/{args.n_splits}")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = create_mlp(X_train.shape[1])
        history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                            validation_data=(X_val, y_val), verbose=0)

        preds = (model.predict(X_val) > 0.5).astype(int).reshape(-1)
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        cm = confusion_matrix(y_val, preds).tolist()
        logger.info(f"Fold {fold} — acc: {acc:.4f}, prec: {prec:.4f}, rec: {rec:.4f}, f1: {f1:.4f}")

        metrics.append({"fold": fold, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm})

        if f1 > best_f1:
            best_f1 = f1
            model.save(best_model_path)
            logger.info(f"Saved best model to {best_model_path} (f1={best_f1:.4f})")

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(args.output_dir, "cv_metrics.csv"), index=False)
    summary = metrics_df[['acc','prec','rec','f1']].mean().to_dict()
    save_json({k: float(v) for k,v in summary.items()}, os.path.join(args.output_dir, "cv_summary.json"))
    logger.info("Training complete.")
    logger.info(f"Cross-validated mean metrics: {summary}")
    logger.info(f"Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main()
