from __future__ import annotations

import numpy as np

from .preprocess import FEATURE_DIM, TOTAL_POINTS


def random_scale(seq: np.ndarray, scale_range: float = 0.1) -> np.ndarray:
    factor = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
    return seq * factor


def random_rotation_xy(seq: np.ndarray, max_degrees: float = 15.0) -> np.ndarray:
    points = seq.reshape(seq.shape[0], TOTAL_POINTS, 3).copy()
    ax = np.deg2rad(np.random.uniform(-max_degrees, max_degrees))
    ay = np.deg2rad(np.random.uniform(-max_degrees, max_degrees))
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rot = ry @ rx
    points = points @ rot.T
    return points.reshape(seq.shape[0], FEATURE_DIM)


def temporal_jitter(seq: np.ndarray, ratio: float = 0.2) -> np.ndarray:
    t = seq.shape[0]
    factor = np.random.uniform(1 - ratio, 1 + ratio)
    new_t = max(4, int(t * factor))
    old_idx = np.linspace(0, 1, t)
    new_idx = np.linspace(0, 1, new_t)
    warped = np.stack([np.interp(new_idx, old_idx, seq[:, d]) for d in range(seq.shape[1])], axis=1)
    out_idx = np.linspace(0, 1, t)
    return np.stack([np.interp(out_idx, np.linspace(0, 1, new_t), warped[:, d]) for d in range(seq.shape[1])], axis=1).astype(np.float32)


def add_gaussian_noise(seq: np.ndarray, sigma: float = 0.005) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, size=seq.shape).astype(np.float32)
    return seq + noise


def augment_sequence(seq: np.ndarray) -> np.ndarray:
    out = random_rotation_xy(seq)
    out = random_scale(out)
    out = temporal_jitter(out)
    out = add_gaussian_noise(out)
    return out.astype(np.float32)
