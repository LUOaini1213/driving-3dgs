import numpy as np

from d3gs.geometry import quat_to_rotmat
from d3gs.splat import BYTES_PER_SPLAT, C0, decode_splat, encode_splat, select_for_web


def _random_gaussians(n, seed=0):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, 4))
    return dict(means=rng.normal(size=(n, 3)) * 20, log_scales=rng.normal(size=(n, 3)) - 3,
                quats=q * rng.uniform(0.5, 3, size=(n, 1)),              # unnormalised on purpose
                opacity_logit=rng.normal(size=n) * 2, sh0=rng.normal(size=(n, 1, 3)))


def test_splat_round_trip():
    g = _random_gaussians(500)
    data = encode_splat(g["means"], g["log_scales"], g["quats"], g["opacity_logit"], g["sh0"])
    assert len(data) == 500 * BYTES_PER_SPLAT
    d = decode_splat(data)
    np.testing.assert_allclose(d["means"], g["means"], rtol=1e-6, atol=1e-5)
    np.testing.assert_allclose(d["scales"], np.exp(g["log_scales"]), rtol=1e-6)
    rgb = np.clip(g["sh0"][:, 0] * C0 + 0.5, 0, 1)
    np.testing.assert_allclose(d["rgb"], rgb, atol=0.5 / 255 + 1e-9)
    np.testing.assert_allclose(d["alpha"], 1 / (1 + np.exp(-g["opacity_logit"])), atol=0.5 / 255 + 1e-9)
    # rotation survives 8-bit quantisation to within a few degrees, in (w, x, y, z) order
    R0 = quat_to_rotmat(g["quats"])
    R1 = quat_to_rotmat(d["quats"])
    cosang = (np.trace(np.einsum("nij,nkj->nik", R0, R1), axis1=1, axis2=2) - 1) / 2
    assert np.degrees(np.arccos(np.clip(cosang, -1, 1))).max() < 3.0


def test_splat_known_bytes():
    data = encode_splat(np.array([[1.0, 2.0, 3.0]]), np.log([[0.5, 1.0, 2.0]]), np.array([[0.0, 0.0, 0.0, 2.0]]),
                        np.array([0.0]), np.zeros((1, 1, 3)))
    raw = np.frombuffer(data, np.uint8)
    np.testing.assert_array_equal(np.frombuffer(data[:24], "<f4"), [1, 2, 3, 0.5, 1, 2])
    assert list(raw[24:28]) == [128, 128, 128, 128]      # grey (sh0 = 0 -> 0.5), alpha sigmoid(0) = 0.5
    assert list(raw[28:32]) == [128, 128, 128, 255]      # unit quaternion (0,0,0,1) -> z component saturates


def test_select_for_web_threshold_cap_and_order():
    opa = np.array([0.9, 0.01, 0.5, 0.9, 0.3])
    ls = np.log(np.array([[1, 1, 1], [9, 9, 9], [2, 2, 1], [0.1, 0.1, 0.1], [1, 1, 1.0]]))
    idx = select_for_web(opa, ls, max_n=3, min_opacity=0.05)
    assert 1 not in idx                                  # below the opacity floor, however large
    assert list(idx) == [2, 0, 4]                        # scores 2.0, 0.9, 0.3 (idx 3 scores 0.009)
