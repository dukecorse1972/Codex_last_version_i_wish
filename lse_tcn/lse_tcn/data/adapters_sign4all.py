from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .preprocess import FEATURE_DIM, normalize_sequence


class Sign4AllAdapter:
    """Adapter for Sign4all HDF5 keypoints; includes fallback extraction hook."""

    def __init__(self, root: str | Path, label_map_path: str | Path | None = None) -> None:
        self.root = Path(root)
        self.label_map = self._load_label_map(label_map_path)

    def _load_label_map(self, path: str | Path | None) -> dict[str, int]:
        if path is None:
            return {}
        p = Path(path)
        if not p.exists():
            return {}
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)

    def load_samples(self, selected_labels: set[str] | None = None) -> list[dict[str, Any]]:
        if not self.root.exists():
            raise FileNotFoundError(f"Sign4all root missing: {self.root}")
        samples: list[dict[str, Any]] = []
        for file in sorted(self.root.glob("*.h5")):
            with h5py.File(file, "r") as hf:
                if "frames" not in hf:
                    continue
                frames = np.asarray(hf["frames"], dtype=np.float32)
                if frames.ndim != 2 or frames.shape[1] != FEATURE_DIM:
                    continue
                label = hf.attrs.get("label", "UNKNOWN")
                label = label.decode() if isinstance(label, bytes) else str(label)
                if selected_labels and label not in selected_labels:
                    continue
                sample = {
                    "frames": normalize_sequence(frames),
                    "label": self.label_map.get(label, -1),
                    "label_name": label,
                    "signer_id": str(hf.attrs.get("signer_id", "unknown")),
                    "source": "sign4all",
                }
                samples.append(sample)
        return samples
