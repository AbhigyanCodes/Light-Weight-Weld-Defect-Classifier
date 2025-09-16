# Quick smoke integration: run feature extraction on a synthetic image saved to disk.
import os
from src.data_utils import extract_features
from PIL import Image
import numpy as np

def test_extract_on_sample(tmp_path):
    arr = (np.random.rand(256,256,3)*255).astype('uint8')
    p = tmp_path / "sample.jpg"
    Image.fromarray(arr).save(str(p))
    feats = extract_features(str(p))
    assert set(['crack_count','crack_length','shear_length']).issubset(set(feats.keys()))
