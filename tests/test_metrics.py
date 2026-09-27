import numpy as np
import pytest

from d3gs.metrics import psnr, psnr_masked, ssim, ssim_masked


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


def test_masked_metrics_equal_full_metrics_when_nothing_is_masked():
    a, b = _img(0), _img(1)
    keep = np.ones(a.shape[:2], bool)
    assert psnr_masked(a, b, keep) == pytest.approx(psnr(a, b), abs=1e-12)
    assert ssim_masked(a, b, keep) == pytest.approx(ssim(a, b), abs=1e-12)


def test_masked_metrics_ignore_errors_inside_the_mask():
    a = _img(0)
    b = a.copy()
    b[20:40, 30:50] = 1 - b[20:40, 30:50]          # corrupt a block (a 'moving object')
    keep = np.ones(a.shape[:2], bool)
    keep[20:40, 30:50] = False
    assert psnr_masked(a, b, keep) == float("inf")
    assert psnr(a, b) < 30                           # the corruption is visible to the full-image metric
    # SSIM windows (radius 5) centred on kept pixels next to the block still see it: only far pixels are exact
    far = keep.copy()
    far[15:45, 25:55] = False
    assert ssim_masked(a, b, far) == pytest.approx(1.0, abs=1e-9)
    assert ssim_masked(a, b, keep) < 1.0
    # PSNR is over kept pixels: a 0.1 error on kept pixels only -> 20 dB
    c = a + 0.1 * keep[..., None]
    assert psnr_masked(a, c, keep) == pytest.approx(20.0)
