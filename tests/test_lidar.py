import numpy as np
import pytest

from d3gs.geometry import intrinsics_matrix, se3
from d3gs.lidar import (LidarLeakError, assert_no_lidar_leak, depth_errors, depth_map, nearest_sweep,
                        split_sweeps)

W, H = 40, 30
K = intrinsics_matrix(20.0, 20.0, 19.5, 14.5)


def test_depth_map_pixel_and_depth_of_known_points():
    # pixel centres at integer coordinates: u = 19.5 + 20 x / z, rounded (no exact .5 ties here)
    pts = np.array([[0.1, 0.1, 10.0],          # (19.7, 14.7) -> (20, 15)
                    [1.075, 0.45, 5.0],        # (23.8, 16.3) -> (24, 16)
                    [-2.1, -1.1, 20.0]])       # (17.4, 13.4) -> (17, 13)
    D = depth_map(K, np.eye(4), pts, W, H)
    ys, xs = np.nonzero(D)
    got = {(int(x), int(y)): float(D[y, x]) for x, y in zip(xs, ys)}
    assert got == pytest.approx({(20, 15): 10.0, (24, 16): 5.0, (17, 13): 20.0})


def test_depth_map_uses_the_camera_pose():
    # camera moved 2 m forward (+z): a point at world z=12 is 10 m away
    cam_SE3_world = np.linalg.inv(se3(np.eye(3), [0, 0, 2.0]))
    D = depth_map(K, cam_SE3_world, np.array([[0.0, 0.0, 12.0]]), W, H)
    assert D.max() == pytest.approx(10.0)


def test_depth_map_keeps_nearest_point_per_pixel():
    pts = np.array([[0.1, 0.1, 30.0], [0.05, 0.05, 15.0], [0.2, 0.2, 60.0]])  # all land on the same pixel
    D = depth_map(K, np.eye(4), pts, W, H)
    assert (D > 0).sum() == 1 and D.max() == pytest.approx(15.0)


def test_depth_map_drops_behind_far_and_out_of_image():
    pts = np.array([[0.0, 0.0, -5.0],        # behind
                    [0.0, 0.0, 0.2],         # nearer than near plane
                    [0.0, 0.0, 200.0],       # beyond far
                    [30.0, 0.0, 10.0],       # far right of the image
                    [0.0, -30.0, 10.0]])     # above the image
    assert not depth_map(K, np.eye(4), pts, W, H).any()


def test_depth_errors_on_known_values():
    gt = np.zeros((2, 3), np.float32)
    gt[0, 0], gt[0, 1], gt[1, 2] = 10.0, 20.0, 5.0
    pred = np.full((2, 3), 99.0)
    pred[0, 0], pred[0, 1], pred[1, 2] = 10.2, 21.0, 5.0
    e = depth_errors(pred, gt)
    assert e["n"] == 3
    assert e["median_abs_m"] == pytest.approx(0.2)
    assert e["within_tol"] == pytest.approx(2 / 3)
    e2 = depth_errors(pred, gt, valid=np.array([[True, False, False], [False, False, False]]))
    assert e2["n"] == 1 and e2["median_abs_m"] == pytest.approx(0.2)


def test_split_sweeps_reserves_nearest_sweep_of_each_heldout_frame():
    frame_ts = np.arange(16) * 50        # 20 Hz camera
    sweep_ts = np.arange(8) * 100 + 7    # 10 Hz lidar, 7 ns phase
    test = [4, 12]
    train_sw, eval_sw = split_sweeps(sweep_ts, frame_ts, test)
    assert eval_sw == [2, 6]              # 207 is nearest to 200, 607 to 600
    assert set(train_sw).isdisjoint(eval_sw)
    assert sorted(train_sw + eval_sw) == list(range(8))
    # a train frame adjacent to a held-out frame gets the nearest NON-eval sweep
    assert nearest_sweep(sweep_ts, frame_ts[3], train_sw) == 1


def test_leak_guard_raises_on_overlap_and_passes_when_disjoint():
    assert_no_lidar_leak([100, 300], [200, 400])
    with pytest.raises(LidarLeakError):
        assert_no_lidar_leak([100, 200, 300], [200, 400])
