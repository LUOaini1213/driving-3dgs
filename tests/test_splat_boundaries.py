import numpy as np
import pytest

from d3gs.splat import decode_splat, encode_splat, select_for_web


def _params(n=1):
    return dict(means=np.zeros((n, 3)), log_scales=np.zeros((n, 3)),
                quats=np.tile([1., 0, 0, 0], (n, 1)), opacity_logit=np.zeros(n), sh0=np.zeros((n, 1, 3)))


@pytest.mark.parametrize("field,value", [
    ("means", [[np.nan, 0, 0]]), ("means", [[1e100, 0, 0]]),
    ("log_scales", [[1000., 0, 0]]), ("log_scales", [[-1000., 0, 0]]),
    ("quats", [[0., 0, 0, 0]]), ("quats", [[1., 0, 0, np.inf]]),
    ("opacity_logit", [np.nan]), ("sh0", [[[np.inf, 0, 0]]]),
    ("means", [[1., 2.]]), ("log_scales", [[0., 0, 0], [0, 0, 0]]),
])
def test_encoder_rejects_invalid_or_unrepresentable_records(field, value):
    args = _params(); args[field] = np.array(value)
    with pytest.raises(ValueError):
        encode_splat(**args)


def test_extreme_valid_logits_and_quaternions_are_encoded_without_warnings():
    args = _params(4)
    args['opacity_logit'] = np.array([-np.inf, -1000., 1000., np.inf])
    args['quats'] *= 1e300
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        decoded = decode_splat(encode_splat(**args))
    np.testing.assert_array_equal(decoded['alpha'], [0, 0, 1, 1])
    np.testing.assert_allclose(decoded['quats'], np.tile([1, 0, 0, 0], (4, 1)))


@pytest.mark.parametrize("offset,raw", [
    (0, np.array([np.inf], '<f4').tobytes()),
    (12, np.array([-1.], '<f4').tobytes()),
    (12, np.array([0.], '<f4').tobytes()),
    (28, bytes([128]*4)),
])
def test_decoder_rejects_corrupt_full_length_record(offset, raw):
    data = bytearray(encode_splat(**_params()))
    data[offset:offset+len(raw)] = raw
    with pytest.raises(ValueError):
        decode_splat(data)


@pytest.mark.parametrize("max_n,min_opacity", [(-1, .1), (True, .1), (1.5, .1), (1, np.nan), (1, -.1), (1, 1.1)])
def test_selection_does_not_silently_reinterpret_invalid_limits(max_n, min_opacity):
    with pytest.raises(ValueError):
        select_for_web(np.array([.5]), np.zeros((1, 3)), max_n, min_opacity)


def test_extreme_scale_ranking_uses_the_same_score_without_overflow_ties():
    # Both direct exp-area scores overflow; their mathematical order is still unambiguous.
    scales = np.array([[0., 500., 500.], [0., 501., 501.]])
    with np.errstate(over='raise'):
        actual = select_for_web(np.array([.5, .5]), scales, 2, 0)
    np.testing.assert_array_equal(actual, [1, 0])


def test_low_level_empty_asset_and_zero_cap_remain_well_defined():
    assert encode_splat(**_params(0)) == b''
    assert decode_splat(b'')['means'].shape == (0, 3)
    assert select_for_web(np.array([.5]), np.zeros((1, 3)), 0, 0).size == 0
    with pytest.raises(ValueError):
        decode_splat(b'\0' * 31)
