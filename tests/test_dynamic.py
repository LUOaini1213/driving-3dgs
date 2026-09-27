import numpy as np
import pandas as pd
import pytest

from d3gs.dynamic import build_tracks, cuboid_corners, cuboid_mask, frame_masks, objects_at, track_speed
from d3gs.geometry import intrinsics_matrix, quat_to_rotmat, se3

W, H = 200, 100
K = intrinsics_matrix(100.0, 100.0, 99.5, 49.5)   # principal point at the image centre


def test_cuboid_corners_extent():
    c = cuboid_corners((4.0, 2.0, 1.0))
    np.testing.assert_allclose(c.min(0), [-2, -1, -0.5])
    np.testing.assert_allclose(c.max(0), [2, 1, 0.5])
    assert len(np.unique(c, axis=0)) == 8


def test_box_in_front_projects_to_expected_rectangle():
    # box centred 10 m ahead, 2 m cube, axes aligned with the camera: near face at z = 9
    m = cuboid_mask(K, se3(np.eye(3), [0, 0, 10.0]), (2.0, 2.0, 2.0), W, H)
    cols = np.where(m.any(0))[0]
    rows = np.where(m.any(1))[0]
    # near face spans u = 99.5 +- 100/9 = [88.39, 110.61], v = 49.5 +- 11.11 = [38.39, 60.61]
    assert cols.min() == 89 and cols.max() == 110
    assert rows.min() == 39 and rows.max() == 60
    assert m.sum() == pytest.approx(22 * 22, abs=0)


def test_box_axes_follow_the_pose():
    # a long thin box (length 6 along object x) rotated so object x = camera y (vertical in the image)
    R = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])               # obj x -> cam y, proper rotation
    assert np.linalg.det(R) == pytest.approx(1.0)
    m = cuboid_mask(K, se3(R, [0, 0, 20.0]), (6.0, 0.4, 0.4), W, H)
    rows = np.where(m.any(1))[0]
    cols = np.where(m.any(0))[0]
    assert rows.max() - rows.min() > 5 * (cols.max() - cols.min())   # tall and thin


def test_box_behind_camera_is_empty():
    m = cuboid_mask(K, se3(np.eye(3), [0, 0, -10.0]), (2.0, 2.0, 2.0), W, H)
    assert not m.any()


def test_box_to_the_left_stays_in_left_half():
    m = cuboid_mask(K, se3(np.eye(3), [-6.0, 0, 10.0]), (2.0, 2.0, 2.0), W, H)
    assert m.any() and not m[:, W // 2:].any()


def test_box_straddling_the_camera_covers_the_image():
    # camera sits inside a 4 m cube: after near-plane clipping every pixel is covered.
    # Without clipping, corners behind the camera project mirrored and the hull is wrong.
    m = cuboid_mask(K, se3(np.eye(3), [0, 0, 0.5]), (4.0, 4.0, 4.0), W, H)
    assert m.all()


def test_box_partly_behind_camera_to_the_right():
    # right-hand box from z=-3 to z=+5, x in [1, 3]: visible part is only on the right side
    m = cuboid_mask(K, se3(np.eye(3), [2.0, 0, 1.0]), (2.0, 2.0, 8.0), W, H)
    assert m.any()
    assert not m[:, : W // 2 + 10].any()      # near edge at u = 99.5 + 100*1/5 = 119.5


def test_track_speed_constant_velocity_and_static():
    ts = (np.arange(30) * 100_000_000).astype(np.int64)      # 10 Hz for 3 s
    moving = np.stack([5.0 * ts / 1e9, np.zeros(30), np.zeros(30)], 1)
    static = np.zeros((30, 3)) + np.random.default_rng(0).normal(scale=0.02, size=(30, 3))
    np.testing.assert_allclose(track_speed(ts, moving), 5.0, atol=1e-9)
    assert track_speed(ts, static).max() < 0.2


class _IdentityPoses:
    def city_SE3_ego(self, t):
        return np.eye(4)


def _ann(xs, ts, cat="REGULAR_VEHICLE", uuid="a"):
    n = len(ts)
    return pd.DataFrame({"timestamp_ns": ts, "track_uuid": [uuid] * n, "category": [cat] * n,
                         "length_m": 4.0, "width_m": 2.0, "height_m": 1.5, "qw": 1.0, "qx": 0.0, "qy": 0.0,
                         "qz": 0.0, "tx_m": xs, "ty_m": 0.0, "tz_m": 0.0})


def test_objects_at_interpolates_between_annotations():
    ts = np.array([0, 100_000_000, 200_000_000], dtype=np.int64)
    tracks = build_tracks(_ann([0.0, 1.0, 2.0], ts), _IdentityPoses())
    (tr, T, dims, sp), = objects_at(tracks, 25_000_000)
    np.testing.assert_allclose(T[:3, 3], [0.25, 0, 0], atol=1e-12)
    assert sp == pytest.approx(10.0)
    assert objects_at(tracks, 400_000_000) == []          # far outside the track: absent


def test_frame_masks_separate_moving_from_parked():
    ts = (np.arange(11) * 100_000_000).astype(np.int64)
    # ego/city frame = camera looking along +x? use a camera whose z axis is city +x
    cam_R = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0.0]])   # columns: cam x,y,z in city (x fwd, y left, z up)
    city_SE3_cam = se3(cam_R, [0, 0, 0])
    moving = _ann(15 + 3.0 * ts / 1e9, ts, uuid="m")             # 3 m/s, straight ahead
    parked = _ann(np.full(11, 12.0), ts, uuid="p")
    parked["ty_m"] = 6.0                                          # 6 m to the left, static
    tracks = build_tracks(pd.concat([moving, parked]), _IdentityPoses())
    mov, veh, n = frame_masks(tracks, 500_000_000, K, city_SE3_cam, W, H)
    assert n == 1 and mov.any()
    assert not mov[:, : W // 4].any()          # the parked car on the left is not in the moving mask
    assert veh[:, : W // 4].any()              # ... but is in the all-vehicles mask
    assert (veh | mov).sum() == veh.sum()      # moving mask is a subset of the vehicle mask


def test_rotated_box_fills_a_diamond_not_its_bounding_box():
    # thin 2 m x 2 m plate at 10 m, rotated 45 deg about the optical axis: its projection is a diamond
    # (diagonals 2*sqrt(2) m -> 28.3 px, area 400 px) whose bounding box is twice as large
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    m = cuboid_mask(K, se3(R, [0, 0, 10.0]), (2.0, 2.0, 0.02), W, H)
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    bbox = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
    assert m.sum() == pytest.approx(400, rel=0.1)
    assert m.sum() < 0.6 * bbox
    assert not m[rows.min(), cols.min()] and not m[rows.max(), cols.max()]   # bbox corners are outside the diamond
