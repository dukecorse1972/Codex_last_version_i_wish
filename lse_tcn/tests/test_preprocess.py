import numpy as np

from lse_tcn.data.preprocess import HAND_LANDMARKS, normalize_frame


def test_normalize_frame_pivot_and_scale() -> None:
    frame = np.zeros((48, 3), dtype=np.float32)
    ls_idx = HAND_LANDMARKS * 2
    rs_idx = ls_idx + 1
    frame[ls_idx] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    frame[rs_idx] = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    frame[0] = np.array([1.0, 1.0, 0.0], dtype=np.float32)

    out = normalize_frame(frame.reshape(-1))
    pts = out.reshape(48, 3)

    assert np.allclose(pts[ls_idx, 0], -0.5, atol=1e-5)
    assert np.allclose(pts[rs_idx, 0], 0.5, atol=1e-5)
    assert np.allclose(pts[0], np.array([0.0, 0.5, 0.0]), atol=1e-5)
