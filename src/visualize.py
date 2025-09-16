"""
visualize.py
Create pairplot, heatmap, boxplots, cv metric bar plot, and confusion matrix heatmap.
Saves PNGs to out-dir.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--cv-metrics", required=False)
    p.add_argument("--predictions", required=False)
    p.add_argument("--out-dir", default="docs/figures")
    return p.parse_args()

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def main():
    args = parse_args()
    out = ensure_dir(args.out_dir)
    feats = pd.read_csv(args.features)

    sns.set(style="whitegrid")
    # Pairplot
    pair_cols = ['crack_count','crack_length','shear_length','label']
    sns.pairplot(feats[pair_cols], hue='label', diag_kind='hist', plot_kws={'alpha':0.6})
    plt.suptitle("Pairplot of features by label", y=1.02)
    plt.savefig(os.path.join(out, "pairplot_features.png"), bbox_inches='tight')
    plt.clf()

    # Correlation heatmap
    corr = feats[['crack_count','crack_length','shear_length']].corr()
    plt.figure(figsize=(6,5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Feature correlation heatmap")
    plt.savefig(os.path.join(out, "correlation_heatmap.png"), bbox_inches='tight')
    plt.clf()

    # Boxplots
    plt.figure(figsize=(6,4))
    sns.boxplot(x='label', y='crack_length', data=feats)
    plt.title("Boxplot: crack_length by label")
    plt.savefig(os.path.join(out, "boxplot_crack_length.png"), bbox_inches='tight')
    plt.clf()

    plt.figure(figsize=(6,4))
    sns.boxplot(x='label', y='crack_count', data=feats)
    plt.title("Boxplot: crack_count by label")
    plt.savefig(os.path.join(out, "boxplot_crack_count.png"), bbox_inches='tight')
    plt.clf()

    # CV metrics plot
    if args.cv_metrics and os.path.exists(args.cv_metrics):
        cv = pd.read_csv(args.cv_metrics)
        mean_scores = cv[['acc','prec','rec','f1']].mean()
        mean_scores.plot(kind='bar', ylim=(0,1))
        plt.title("Cross-validated mean metrics")
        plt.ylabel("Score")
        plt.savefig(os.path.join(out, "cv_mean_metrics.png"), bbox_inches='tight')
        plt.clf()

    # Confusion matrix from predictions
    if args.predictions and os.path.exists(args.predictions):
        preds = pd.read_csv(args.predictions)
        if 'label' in preds.columns and 'pred_label' in preds.columns:
            cm = confusion_matrix(preds['label'], preds['pred_label'])
            plt.figure(figsize=(5,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(out, "confusion_matrix.png"), bbox_inches='tight')
            plt.clf()

    print(f"Saved figures to {out}")

if __name__ == "__main__":
    main()
