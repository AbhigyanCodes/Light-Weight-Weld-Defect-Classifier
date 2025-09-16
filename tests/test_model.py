from src.model import create_mlp
def test_mlp_shape():
    m = create_mlp(3)
    # check can produce prediction for a dummy input
    import numpy as np
    out = m.predict(np.zeros((1,3)))
    assert out.shape == (1,1)
