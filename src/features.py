"""
features.py
A thin wrapper for feature extraction utilities which can be extended for advanced segmentation.
"""

from src.data_utils import extract_features, build_feature_dataframe

# Re-export
__all__ = ["extract_features", "build_feature_dataframe"]
