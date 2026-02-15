from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .preprocess import FEATURE_DIM, normalize_sequence


class SWLLSEAdapter:
    """Adapter for SWL-LSE pkl files with precomputed MediaPipe outputs."""

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
            raise FileNotFoundError(f"SWL-LSE root missing: {self.root}")
        samples: list[dict[str, Any]] = []
        for file in sorted(self.root.glob("*.pkl")):
            with file.open("rb") as f:
                data = pickle.load(f)
            frames = np.asarray(data.get("frames"), dtype=np.float32)
            if frames.ndim != 2 or frames.shape[1] != FEATURE_DIM:
                continue
            label = str(data.get("label", "UNKNOWN"))
            if selected_labels and label not in selected_labels:
                continue
            sample = {
                "frames": normalize_sequence(frames),
                "label": self.label_map.get(label, -1),
                "label_name": label,
                "signer_id": str(data.get("signer_id", "unknown")),
                "source": "swl_lse",
            }
            samples.append(sample)
        return samples
