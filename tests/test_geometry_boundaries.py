import numpy as np
import pytest

from d3gs.dynamic import cuboid_mask, frame_masks, track_speed
from d3gs.geometry import (interpolate_pose, intrinsics_matrix, project, quat_to_rotmat,
                           resize_intrinsics, se3_from_quat_trans, slerp)
from d3gs.lidar import depth_map


@pytest.mark.parametrize("q", [[0, 0, 0, 0], [1, 0, 0, np.nan], [1, 0, 0, np.inf], [1, 0, 0, 0, 1]])
def test_invalid_rotation_cannot_become_an_empty_moving_mask(q):
    with pytest.raises(ValueError):
        se3_from_quat_trans(q, [0, 0, 10])
    with pytest.raises(ValueError):
        slerp(q, [1, 0, 0, 0], .5)


@pytest.mark.parametrize("scale", [1e-300, 1e300])
def test_quaternion_normalisation_is_scale_invariant(scale):
    q = np.array([1., 0., 0., 1.])
    np.testing.assert_allclose(quat_to_rotmat(q * scale), quat_to_rotmat(q), atol=1e-15)


def test_skew_intrinsics_resize_commutes_with_pixel_mapping():
    K = np.array([[100., 20., 99.5], [3., 100., 49.5], [0., 0., 1.]])
    pts = np.array([[1., 2., 10.], [-2., .5, 4.]])
    original, _ = project(K, np.eye(4), pts)
    resized, _ = project(resize_intrinsics(K, (200, 100), (100, 75)), np.eye(4), pts)
    np.testing.assert_allclose(resized, (original + .5) * [.5, .75] - .5)


@pytest.mark.parametrize("which", ["pose", "intrinsics", "dimensions", "near", "width"])
def test_invalid_cuboid_inputs_fail_before_behind_camera_empty_return(which):
    K = intrinsics_matrix(100, 100, 20, 20)
    T = np.eye(4); T[2, 3] = -10
    dims, near, width = [2., 2., 2.], .1, 40
    if which == "pose": T[0, 0] = np.nan
    if which == "intrinsics": K[0, 0] = np.nan
    if which == "dimensions": dims[0] = -2
    if which == "near": near = 0
    if which == "width": width = True
    with pytest.raises(ValueError):
        cuboid_mask(K, T, dims, width, 40, near)


@pytest.mark.parametrize("field,value", [("speed_thresh", np.nan), ("pad_m", -.5)])
def test_empty_tracks_do_not_hide_invalid_mask_options(field, value):
    with pytest.raises(ValueError):
        frame_masks([], 0, np.eye(3), np.eye(4), 20, 20, **{field: value})


@pytest.mark.parametrize("ts", [[], [0, 0], [1, 0], [0., 1.5]])
def test_pose_timestamps_are_nonempty_unique_ordered_integer_ns(ts):
    with pytest.raises(ValueError):
        interpolate_pose(np.array(ts), np.tile([1., 0, 0, 0], (len(ts), 1)), np.zeros((len(ts), 3)), 0)


def test_epoch_nanoseconds_do_not_change_track_speed():
    # At epoch-sized ns, converting absolute timestamps to float seconds loses small differences.
    ts = np.array([0, 100, 200], np.int64)
    centres = np.array([[0., 0, 0], [1e-7, 0, 0], [2e-7, 0, 0]])
    np.testing.assert_allclose(track_speed(ts + 1_700_000_000_000_000_000, centres, 1e-7),
                               [1., 1., 1.], atol=1e-12)


@pytest.mark.parametrize("kwargs", [{"W": 0}, {"near": 0}, {"near": 3, "far": 2}])
def test_depth_projection_rejects_invalid_domain_with_empty_points(kwargs):
    args = dict(K=np.eye(3), cam_SE3_world=np.eye(4), pts_world=np.empty((0, 3)), W=20, H=20)
    args.update(kwargs)
    with pytest.raises(ValueError):
        depth_map(**args)
