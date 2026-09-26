"""Minimal readers for the Argoverse 2 sensor-log layout (no av2 package needed)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import intrinsics_matrix, interpolate_pose, se3_from_quat_trans


@dataclass
class CameraCalib:
    name: str
    K: np.ndarray            # 3x3, full-resolution pixels
    dist: np.ndarray         # (k1, k2, k3) radial
    width: int
    height: int
    ego_SE3_cam: np.ndarray  # 4x4


@dataclass
class PoseTable:
    ts: np.ndarray      # (N,) int64 ns, sorted
    quats: np.ndarray   # (N, 4) qw qx qy qz
    trans: np.ndarray   # (N, 3)

    def city_SE3_ego(self, t_ns: int) -> np.ndarray:
        return interpolate_pose(self.ts, self.quats, self.trans, int(t_ns))

    def gap_ns(self, t_ns: int) -> int:
        """Distance to the nearest stored pose sample (0 = exact match)."""
        i = int(np.argmin(np.abs(self.ts - int(t_ns))))
        return int(abs(self.ts[i] - int(t_ns)))


def load_camera_calib(log_dir: Path, camera: str) -> CameraCalib:
    intr = pd.read_feather(log_dir / "calibration" / "intrinsics.feather")
    extr = pd.read_feather(log_dir / "calibration" / "egovehicle_SE3_sensor.feather")
    r = intr[intr.sensor_name == camera].iloc[0]
    e = extr[extr.sensor_name == camera].iloc[0]
    return CameraCalib(
        name=camera,
        K=intrinsics_matrix(r.fx_px, r.fy_px, r.cx_px, r.cy_px),
        dist=np.array([r.k1, r.k2, r.k3], dtype=np.float64),
        width=int(r.width_px),
        height=int(r.height_px),
        ego_SE3_cam=se3_from_quat_trans(e[["qw", "qx", "qy", "qz"]].to_numpy(float), e[["tx_m", "ty_m", "tz_m"]].to_numpy(float)),
    )


def load_poses(log_dir: Path) -> PoseTable:
    p = pd.read_feather(log_dir / "city_SE3_egovehicle.feather").sort_values("timestamp_ns")
    return PoseTable(
        ts=p.timestamp_ns.to_numpy(np.int64),
        quats=p[["qw", "qx", "qy", "qz"]].to_numpy(float),
        trans=p[["tx_m", "ty_m", "tz_m"]].to_numpy(float),
    )


def list_frames(log_dir: Path, camera: str) -> list[tuple[int, Path]]:
    fs = sorted((log_dir / "sensors" / "cameras" / camera).glob("*.jpg"), key=lambda p: int(p.stem))
    return [(int(p.stem), p) for p in fs]


def list_sweeps(log_dir: Path) -> list[tuple[int, Path]]:
    fs = sorted((log_dir / "sensors" / "lidar").glob("*.feather"), key=lambda p: int(p.stem))
    return [(int(p.stem), p) for p in fs]


def load_sweep_xyz(path: Path) -> np.ndarray:
    d = pd.read_feather(path, columns=["x", "y", "z"])
    return d.to_numpy(np.float64)
