"""The prepared-scene accessors refuse to hand held-out lidar to training code (and vice versa)."""
import json

import numpy as np
import pytest

from d3gs.scene import Scene


@pytest.fixture()
def work(tmp_path):
    frames = [{"idx": i, "timestamp_ns": i * 50_000_000, "image": f"images/{i:04d}.png",
               "c2w": np.eye(4).tolist()} for i in range(4)]
    frames[1]["c2w"][0][3] = 1.0
    (tmp_path / "cameras.json").write_text(json.dumps({
        "width": 8, "height": 6, "K": np.eye(3).tolist(), "frames": frames, "train": [0, 1, 3], "test": [2]}))
    (tmp_path / "depth").mkdir()
    (tmp_path / "eval_depth").mkdir()
    for i in (0, 1, 3):
        np.save(tmp_path / "depth" / f"{i:04d}.npy", np.full((6, 8), float(i), np.float32))
    np.save(tmp_path / "eval_depth" / "0002.npy", np.full((6, 8), 2.0, np.float32))
    # a stray file that must never be served as training depth
    np.save(tmp_path / "depth" / "0002.npy", np.full((6, 8), 99.0, np.float32))
    return tmp_path


def test_train_depth_only_for_train_frames(work):
    sc = Scene(work)
    assert sc.train_depth(1)[0, 0] == 1.0
    with pytest.raises(PermissionError):
        sc.train_depth(2)


def test_eval_depth_only_for_heldout_frames(work):
    sc = Scene(work)
    assert sc.eval_depth(2)[0, 0] == 2.0
    with pytest.raises(PermissionError):
        sc.eval_depth(0)
