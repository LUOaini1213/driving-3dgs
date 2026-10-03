"""Shared torch-side helpers: load prepared scene, build/render Gaussians, SSIM loss."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

C0 = 0.28209479177387814  # SH band-0 constant


def rgb_to_sh0(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


def sh0_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    return sh0 * C0 + 0.5


class Scene:
    def __init__(self, work_dir: str | Path):
        self.dir = Path(work_dir)
        self.meta = json.loads((self.dir / "cameras.json").read_text(encoding="utf-8"))
        self._validate_metadata()
        self.W, self.H = self.meta["width"], self.meta["height"]
        self.K = torch.tensor(self.meta["K"], dtype=torch.float32)
        self.c2w = torch.tensor(np.array([f["c2w"] for f in self.meta["frames"]]), dtype=torch.float32)
        self.train = list(self.meta["train"])
        self.train_stride = int(self.meta.get("train_stride", 1))
        self.test = list(self.meta["test"])
        centers = self.c2w[self.train, :3, 3]
        self.scene_scale = max(1e-3, float((centers - centers.mean(0)).norm(dim=1).max()) * 1.1)

    def _validate_metadata(self):
        m = self.meta
        if not isinstance(m, dict):
            raise ValueError("scene metadata must be an object")
        for key in ("width", "height", "train_stride"):
            value = m.get(key, 1 if key == "train_stride" else None)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        try:
            K = np.asarray(m["K"], dtype=float)
            if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
                raise ValueError("invalid camera K")
            if abs(np.linalg.det(K)) < 1e-12:
                raise ValueError("singular camera K")
            frames = m["frames"]
            if not isinstance(frames, list) or not frames:
                raise ValueError("frames must be nonempty")
            timestamps = []
            for i, frame in enumerate(frames):
                if frame.get("idx", i) != i or type(frame.get("idx", i)) is not int:
                    raise ValueError("frame idx must match its position")
                timestamp = frame["timestamp_ns"]
                if type(timestamp) is not int:
                    raise ValueError("frame timestamps must be integers")
                timestamps.append(timestamp)
                pose = np.asarray(frame["c2w"], dtype=float)
                if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
                    raise ValueError("invalid camera pose")
                R = pose[:3, :3]
                if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(R), 1, atol=1e-5):
                    raise ValueError("camera pose rotation must be rigid")
                name = frame["image"]
                if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
                    raise ValueError("image must be a relative path inside the scene")
                if not (self.dir / name).resolve().is_relative_to(self.dir.resolve()):
                    raise ValueError("image escapes prepared scene")
            if any(a >= b for a, b in zip(timestamps, timestamps[1:])):
                raise ValueError("frame timestamps must be strictly increasing")
            for key in ("train", "test"):
                ids = m[key]
                if not isinstance(ids, list) or any(type(i) is not int or not 0 <= i < len(frames) for i in ids) or len(ids) != len(set(ids)):
                    raise ValueError(f"invalid {key} frame indices")
            if not m["train"] or set(m["train"]) & set(m["test"]):
                raise ValueError("train must be nonempty and disjoint from held-out frames")
            if "city_origin" in m:
                origin = np.asarray(m["city_origin"], dtype=float)
                if origin.shape != (3,) or not np.isfinite(origin).all():
                    raise ValueError("city_origin must be a finite 3-vector")
        except (KeyError, TypeError) as exc:
            raise ValueError("incomplete scene metadata") from exc

    def image(self, i: int) -> np.ndarray:
        """uint8 RGB (H, W, 3)."""
        import cv2
        from .image_io import read_image
        image = read_image(self.dir / self.meta["frames"][i]["image"], cv2.IMREAD_COLOR)
        if image.shape != (self.H, self.W, 3):
            raise ValueError(f"image {i} shape does not match scene resolution")
        return image[:, :, ::-1].copy()

    def initial_points(self):
        with np.load(self.dir / "init_points.npz", allow_pickle=False) as data:
            xyz, rgb = data["xyz"], data["rgb"]
        if (xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(xyz) < 4 or rgb.shape != xyz.shape
                or not np.isfinite(xyz).all() or not np.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any()):
            raise ValueError("initial points need >=4 finite xyz/RGB rows, with RGB in [0,1]")
        return xyz, rgb

    def viewmat(self, i: int) -> torch.Tensor:
        return torch.linalg.inv(self.c2w[i])

    def mask(self, i: int, kind: str = "moving") -> np.ndarray:
        """bool (H, W): True = pixel covered by a moving object ('moving') or any vehicle ('vehicles')."""
        import cv2
        p = self.dir / "masks" / kind / f"{i:04d}.png"
        if not p.exists():
            raise FileNotFoundError(f"{p} (run prepare.py with --masks)")
        from .image_io import read_image
        mask = read_image(p, cv2.IMREAD_GRAYSCALE)
        if mask.shape != (self.H, self.W):
            raise ValueError(f"mask {i} shape does not match scene resolution")
        return mask > 0

    @property
    def lidar_split(self) -> dict | None:
        p = self.dir / "lidar_split.json"
        if not p.exists():
            return None
        split = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(split, dict) or split.get("policy") not in ("strict", "legacy"):
            raise ValueError("invalid lidar split policy")
        for key in ("sweep_ts", "eval_sweep_ts", "init_sweep_ts", "depth_sweep_ts"):
            values = split.get(key)
            if not isinstance(values, list) or any(type(t) is not int for t in values) or len(values) != len(set(values)):
                raise ValueError(f"invalid lidar {key}")
        sweeps = split["sweep_ts"]
        if not sweeps or sweeps != sorted(sweeps):
            raise ValueError("lidar sweep timestamps must be nonempty and sorted")
        for key in ("eval_sweep_ts", "init_sweep_ts", "depth_sweep_ts"):
            if not set(split[key]) <= set(sweeps):
                raise ValueError(f"unknown lidar timestamps in {key}")
        from .lidar import split_sweeps, assert_no_lidar_leak
        _, expected = split_sweeps(np.asarray(sweeps), np.asarray([f["timestamp_ns"] for f in self.meta["frames"]]), self.test)
        if set(split["eval_sweep_ts"]) != {sweeps[k] for k in expected}:
            raise ValueError("lidar evaluation split does not match held-out frames")
        if split["policy"] == "strict":
            assert_no_lidar_leak(split["init_sweep_ts"] + split["depth_sweep_ts"], split["eval_sweep_ts"])
        return split

    def _depth(self, path):
        depth = np.load(path, allow_pickle=False)
        if depth.shape != (self.H, self.W) or not np.isfinite(depth).all() or (depth < 0).any():
            raise ValueError(f"invalid depth data: {path}")
        return depth

    def train_depth(self, i: int) -> np.ndarray:
        """Sparse lidar z-depth (H, W) for a TRAIN frame (0 = no return). Refuses held-out frames."""
        if i not in self.train:
            raise PermissionError(f"frame {i} is not a train frame; its lidar is reserved for evaluation")
        return self._depth(self.dir / "depth" / f"{i:04d}.npy")

    def eval_depth(self, i: int) -> np.ndarray:
        """Sparse lidar depth of a HELD-OUT frame from its reserved evaluation sweep (evaluation only)."""
        if i not in self.test:
            raise PermissionError(f"frame {i} is not held out")
        return self._depth(self.dir / "eval_depth" / f"{i:04d}.npy")


def init_params(xyz: np.ndarray, rgb: np.ndarray, sh_degree: int, device: str) -> torch.nn.ParameterDict:
    pts = torch.tensor(xyz, dtype=torch.float32)
    n = len(pts)
    # initial scale = mean distance to 3 nearest neighbours (chunked, GPU)
    p = pts.to(device)
    d = torch.empty(n, device=device)
    for s in range(0, n, 256):
        dd = torch.cdist(p[s:s + 256], p)
        d[s:s + 256] = dd.topk(4, largest=False).values[:, 1:].mean(1)
    d = d.clamp_min(1e-3).cpu()
    k = (sh_degree + 1) ** 2
    shN = torch.zeros(n, k - 1, 3)
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(pts),
        "scales": torch.nn.Parameter(torch.log(d)[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(torch.tensor([1.0, 0, 0, 0]).repeat(n, 1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1))),
        "sh0": torch.nn.Parameter(rgb_to_sh0(torch.tensor(rgb, dtype=torch.float32))[:, None, :]),
        "shN": torch.nn.Parameter(shN),
    }).to(device)


def render(params, viewmat: torch.Tensor, K: torch.Tensor, W: int, H: int, sh_degree: int, with_depth: bool = False):
    """Returns (rgb (H,W,3), alpha (H,W,1), info) or, with_depth, (rgb, alpha, info, expected z-depth (H,W))."""
    from gsplat import rasterization
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    out, alpha, info = rasterization(
        means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
        opacities=torch.sigmoid(params["opacities"]), colors=colors,
        viewmats=viewmat[None], Ks=K[None], width=W, height=H, sh_degree=sh_degree,
        packed=False, render_mode="RGB+ED" if with_depth else "RGB",
    )
    img = out[0, ..., :3].clamp(0, 1)
    if with_depth:
        return img, alpha[0], info, out[0, ..., 3]
    return img, alpha[0], info


def _gauss_window(size=11, sigma=1.5, device="cpu"):
    x = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None].repeat(3, 1, 1, 1)


def ssim_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM for (H, W, 3) tensors in [0,1] ('same' padding; used only as a training loss)."""
    a = a.permute(2, 0, 1)[None]
    b = b.permute(2, 0, 1)[None]
    w = _gauss_window(device=a.device)
    mu_a = F.conv2d(a, w, padding=5, groups=3)
    mu_b = F.conv2d(b, w, padding=5, groups=3)
    saa = F.conv2d(a * a, w, padding=5, groups=3) - mu_a ** 2
    sbb = F.conv2d(b * b, w, padding=5, groups=3) - mu_b ** 2
    sab = F.conv2d(a * b, w, padding=5, groups=3) - mu_a * mu_b
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return m.mean()


def exp_lr(step: int, max_steps: int, lr0: float, lr1: float) -> float:
    t = min(max(step / max(max_steps, 1), 0.0), 1.0)
    return math.exp((1 - t) * math.log(lr0) + t * math.log(lr1))
