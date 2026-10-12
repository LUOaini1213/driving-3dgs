"""Independent metric boundary/cohort regressions; no renderer or dataset needed."""
import numpy as np
import pytest

from d3gs.lidar import depth_errors
from d3gs.metrics import psnr, psnr_masked, ssim, ssim_masked


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_missing_prediction_cannot_improve_a_fixed_depth_cohort(bad):
    gt = np.array([10.0, 100.0])
    original = depth_errors(np.array([10.0, 0.0]), gt)
    assert original["n"] == 2 and original["mean_abs_m"] == 50
    assert original["within_tol"] == .5
    with pytest.raises(ValueError, match="prediction"):
        depth_errors(np.array([10.0, bad]), gt)


def test_depth_cohort_uses_gt_and_explicit_mask_only():
    result = depth_errors(np.array([10., np.nan, np.inf]), np.array([10., 20., 0.]),
                          np.array([True, False, True]))
    assert result == {"n": 1, "median_abs_m": 0., "mean_abs_m": 0., "within_tol": 1., "abs_rel": 0.}
    for pred, gt in [(np.array([]), np.array([])), (np.array([np.nan]), np.array([0.]))]:
        assert depth_errors(pred, gt) == {
            "n": 0, "median_abs_m": None, "mean_abs_m": None, "within_tol": None, "abs_rel": None}


@pytest.mark.parametrize("kwargs", [
    {"pred": np.zeros(2), "gt": np.ones(1)},
    {"pred": np.zeros(2), "gt": np.ones(2), "valid": np.array([True])},
    {"pred": np.zeros(2), "gt": np.ones(2), "valid": np.array([1., np.nan])},
    {"pred": np.zeros(2), "gt": np.array([1., np.inf])},
    {"pred": np.zeros(2), "gt": np.array([1., -1.])},
    {"pred": np.zeros(2), "gt": np.ones(2), "tol": -1},
    {"pred": np.zeros(2), "gt": np.ones(2), "tol": np.nan},
])
def test_depth_invalid_domain_is_rejected(kwargs):
    with pytest.raises(ValueError):
        depth_errors(**kwargs)


def test_masked_ssim_does_not_ignore_extra_error_channels():
    a = np.zeros((12, 12, 1))
    b = np.zeros((12, 12, 3)); b[..., 1:] = 1
    with pytest.raises(ValueError, match="shape"):
        ssim_masked(a, b, np.ones((12, 12), bool))


@pytest.mark.parametrize("metric", [psnr, ssim, psnr_masked, ssim_masked])
@pytest.mark.parametrize("bad", ["empty", "nan", "inf", "range_zero", "range_nan"])
def test_image_metric_rejects_invalid_images_and_range(metric, bad):
    a = np.zeros((0 if bad == "empty" else 12, 12, 3))
    b = a.copy()
    if bad in ("nan", "inf"):
        b[0, 0, 0] = np.nan if bad == "nan" else np.inf
    kwargs = {"data_range": 0 if bad == "range_zero" else np.nan if bad == "range_nan" else 1}
    if metric in (psnr_masked, ssim_masked):
        kwargs["keep"] = np.ones(a.shape[:2], bool)
    with pytest.raises(ValueError):
        metric(a, b, **kwargs)


@pytest.mark.parametrize("metric", [ssim, ssim_masked])
def test_ssim_requires_one_complete_window(metric):
    a = np.zeros((10, 12, 3))
    args = (a, a) if metric is ssim else (a, a, np.ones(a.shape[:2], bool))
    with pytest.raises(ValueError):
        metric(*args)


@pytest.mark.parametrize("metric", [psnr_masked, ssim_masked])
@pytest.mark.parametrize("mask", [np.zeros((13, 12), bool), np.full((12, 12), np.nan), np.full((12, 12), 2)])
def test_metric_validates_mask_even_when_no_pixels_are_kept(metric, mask):
    a = np.zeros((12, 12, 3))
    with pytest.raises(ValueError):
        metric(a, a, mask)


def test_identity_empty_support_and_ssim_window_contract_remain():
    a = np.full((30, 30, 3), .5); b = a.copy(); b[12:18, 12:18] = 1.
    keep = np.ones((30, 30), bool); keep[12:18, 12:18] = False
    assert psnr(a, a) == np.inf
    assert psnr_masked(a, b, keep) == np.inf
    assert ssim_masked(a, b, keep) < 1  # neighbouring kept windows still see the excluded block
    for metric in (psnr_masked, ssim_masked):
        assert np.isnan(metric(a, b, np.zeros((30, 30), bool)))
    grey = a[..., 0]
    assert ssim_masked(grey, grey, np.ones((30, 30), bool)) == pytest.approx(1)
