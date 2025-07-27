import numpy as np
from src.utils.cwt import compute_scalogram, resize_vertical_all_repeat


def test_compute_scalogram_shape_and_range():
    signal = np.random.randn(2048)
    result = compute_scalogram(signal)
    assert result.shape == (224,)
    assert np.all(result >= 0) and np.all(result <= 1)


def test_resize_vertical_all_repeat():
    img = np.arange(12).reshape(3, 4)
    resized = resize_vertical_all_repeat(img, target_height=5)
    assert resized.shape == (5, 4)
    # rows should repeat in order
    assert np.array_equal(resized[0:3], img)
    assert np.array_equal(resized[3:], img[:2])
