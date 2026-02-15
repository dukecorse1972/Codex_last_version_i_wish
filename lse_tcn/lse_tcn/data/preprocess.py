from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

HAND_LANDMARKS = 21
POSE_IDS = [11, 12, 13, 14, 15, 16]
TOTAL_POINTS = HAND_LANDMARKS * 2 + len(POSE_IDS)
FEATURE_DIM = TOTAL_POINTS * 3


@dataclass
class SkeletonFrame:
    features: np.ndarray  # [144]


def _landmarks_to_array(landmarks: Iterable, expected: int) -> np.ndarray:
    if landmarks is None:
        return np.zeros((expected, 3), dtype=np.float32)
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    if arr.shape[0] < expected:
        padded = np.zeros((expected, 3), dtype=np.float32)
        padded[: arr.shape[0]] = arr
        return padded
    return arr[:expected]


def extract_frame_features(results: object) -> np.ndarray:
    left = _landmarks_to_array(getattr(getattr(results, "left_hand_landmarks", None), "landmark", None), HAND_LANDMARKS)
    right = _landmarks_to_array(getattr(getattr(results, "right_hand_landmarks", None), "landmark", None), HAND_LANDMARKS)
    pose_all = _landmarks_to_array(getattr(getattr(results, "pose_landmarks", None), "landmark", None), 33)
    pose = pose_all[POSE_IDS]
    return np.concatenate([left, right, pose], axis=0).reshape(-1).astype(np.float32)


def normalize_frame(
    frame: np.ndarray,
    rotate_align: bool = False,
    eps: float = 1e-6,
) -> np.ndarray:
    points = frame.reshape(TOTAL_POINTS, 3).copy()
    left_shoulder = points[HAND_LANDMARKS * 2 + 0]
    right_shoulder = points[HAND_LANDMARKS * 2 + 1]
    pivot = (left_shoulder + right_shoulder) / 2.0
    points -= pivot

    shoulder_vec = right_shoulder - left_shoulder
    scale = np.linalg.norm(shoulder_vec) + eps
    points /= scale

    if rotate_align:
        angle = np.arctan2(shoulder_vec[1], shoulder_vec[0])
        c, s = np.cos(-angle), np.sin(-angle)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        points = points @ rot.T

    return points.reshape(-1).astype(np.float32)


def normalize_sequence(sequence: np.ndarray, rotate_align: bool = False) -> np.ndarray:
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected [T, {FEATURE_DIM}] got {sequence.shape}")
    return np.stack([normalize_frame(f, rotate_align=rotate_align) for f in sequence], axis=0)


def movement_magnitude(sequence: np.ndarray) -> float:
    hands = sequence[:, : HAND_LANDMARKS * 2 * 3]
    if hands.shape[0] < 2:
        return 0.0
    deltas = np.diff(hands, axis=0)
    return float(np.mean(np.linalg.norm(deltas.reshape(deltas.shape[0], -1), axis=1)))
