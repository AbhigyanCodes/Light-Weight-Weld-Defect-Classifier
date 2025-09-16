import os
from src.data_utils import basic_preprocess, extract_features

def test_basic_preprocess():
    import numpy as np
    img = (np.random.rand(200,200,3)*255).astype('uint8')
    gray = basic_preprocess(img, size=(128,128))
    assert gray.shape == (128,128)
