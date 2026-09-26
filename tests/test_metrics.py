import numpy as np
import pytest

from d3gs.metrics import psnr, ssim


def _img(seed=0, h=64, w=80):
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 1, w)[None, :, None] * np.ones((h, 1, 3))
    return np.clip(base + 0.1 * rng.normal(size=(h, w, 3)), 0, 1)


def test_psnr_known_mse():
    a = np.zeros((10, 10, 3))
    b = np.full((10, 10, 3), 0.1)  # MSE = 0.01 -> 20 dB
    assert psnr(a, b) == pytest.approx(20.0)
    assert psnr(a, a) == float("inf")


def test_ssim_identity_and_ordering():
    a = _img(0)
    assert ssim(a, a) == pytest.approx(1.0)
    rng = np.random.default_rng(5)
    light = np.clip(a + 0.02 * rng.normal(size=a.shape), 0, 1)
    heavy = np.clip(a + 0.2 * rng.normal(size=a.shape), 0, 1)
    assert 1.0 > ssim(a, light) > ssim(a, heavy) > 0


def test_ssim_matches_scikit_image_reference():
    skm = pytest.importorskip("skimage.metrics")
    a = _img(1)
    b = np.clip(a + 0.05 * np.random.default_rng(2).normal(size=a.shape), 0, 1)
    ref = skm.structural_similarity(a, b, channel_axis=2, data_range=1.0, gaussian_weights=True,
                                    sigma=1.5, use_sample_covariance=False)
    # skimage crops a 5-px border after 'same' filtering; ours uses the 'valid' region -> identical support
    assert ssim(a, b) == pytest.approx(ref, abs=1e-6)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        psnr(np.zeros((4, 4, 3)), np.zeros((4, 5, 3)))
