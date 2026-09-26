import numpy as np
import pytest

from d3gs.geometry import (interpolate_pose, project, quat_to_rotmat, resize_intrinsics, rotmat_to_quat, se3,
                           se3_from_quat_trans, se3_inverse, transform_points)


def test_quat_known_rotation_90deg_about_z():
    q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])  # +90 deg about z
    R = quat_to_rotmat(q)
    np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(R @ [0, 1, 0], [-1, 0, 0], atol=1e-12)


def test_quat_rotmat_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(200):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q = q if q[0] >= 0 else -q
        R = quat_to_rotmat(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0)
        np.testing.assert_allclose(rotmat_to_quat(R), q, atol=1e-9)


def test_se3_known_transform_and_inverse_round_trip():
    # frame b is rotated +90 deg about z and shifted by (10, 0, 2) inside frame a
    T = se3_from_quat_trans([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)], [10.0, 0.0, 2.0])
    p_b = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
    p_a = transform_points(T, p_b)
    np.testing.assert_allclose(p_a, [[10, 1, 2], [10, 0, 7]], atol=1e-12)
    np.testing.assert_allclose(transform_points(se3_inverse(T), p_a), p_b, atol=1e-12)
    np.testing.assert_allclose(se3_inverse(T) @ T, np.eye(4), atol=1e-12)


def test_camera_chain_matches_manual_composition():
    rng = np.random.default_rng(2)
    city_ego = se3(quat_to_rotmat(rng.normal(size=4)), rng.normal(size=3) * 100)
    ego_cam = se3(quat_to_rotmat(rng.normal(size=4)), rng.normal(size=3))
    p_cam = rng.normal(size=(10, 3))
    chained = transform_points(city_ego @ ego_cam, p_cam)
    manual = transform_points(city_ego, transform_points(ego_cam, p_cam))
    np.testing.assert_allclose(chained, manual, atol=1e-9)


def test_interpolate_pose_exact_midpoint_and_range():
    ts = np.array([0, 100], dtype=np.int64)
    qz90 = [np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)]
    quats = np.array([[1, 0, 0, 0], qz90], float)
    trans = np.array([[0, 0, 0], [10, 0, 0]], float)
    np.testing.assert_allclose(interpolate_pose(ts, quats, trans, 100), se3_from_quat_trans(qz90, [10, 0, 0]))
    mid = interpolate_pose(ts, quats, trans, 50)
    np.testing.assert_allclose(mid[:3, 3], [5, 0, 0])
    np.testing.assert_allclose(mid[:3, :3], quat_to_rotmat([np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)]), atol=1e-12)
    # asymmetric query: catches a reversed interpolation weight (the midpoint alone cannot)
    q25 = interpolate_pose(ts, quats, trans, 25)
    np.testing.assert_allclose(q25[:3, 3], [2.5, 0, 0])
    np.testing.assert_allclose(q25[:3, :3], quat_to_rotmat([np.cos(np.pi / 16), 0, 0, np.sin(np.pi / 16)]), atol=1e-12)
    with pytest.raises(ValueError):
        interpolate_pose(ts, quats, trans, 101)


def test_resize_intrinsics_maps_pixel_centres():
    K = np.array([[1000.0, 0, 775.5], [0, 1000.0, 1023.0], [0, 0, 1]])
    K2 = resize_intrinsics(K, (1550, 2048), (388, 512))
    # a 3-D point projecting to full-res pixel (u, v) must project to s*(u+0.5)-0.5 after resizing
    X = np.array([[0.3, -0.2, 4.0]])
    uv1, _ = project(K, np.eye(4), X)
    uv2, _ = project(K2, np.eye(4), X)
    sx, sy = 388 / 1550, 512 / 2048
    np.testing.assert_allclose(uv2[0], [sx * (uv1[0, 0] + 0.5) - 0.5, sy * (uv1[0, 1] + 0.5) - 0.5], atol=1e-9)
    # image-corner convention: the full-res right edge (u = W - 0.5) maps to the small right edge
    assert sx * (1550 - 0.5 + 0.5) - 0.5 == pytest.approx(388 - 0.5)
