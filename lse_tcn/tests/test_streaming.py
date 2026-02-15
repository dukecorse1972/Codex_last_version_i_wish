import numpy as np

from lse_tcn.realtime_demo import StreamingState


def test_streaming_buffer_pipeline() -> None:
    s = StreamingState(window_size=6, infer_stride=2, smooth_window=3)
    for _ in range(6):
        s.push(np.zeros(144, dtype=np.float32))
    assert s.ready()
    win = s.get_window()
    assert win.shape == (6, 144)
    p1 = s.smooth(np.array([0.2, 0.8], dtype=np.float32))
    p2 = s.smooth(np.array([0.6, 0.4], dtype=np.float32))
    assert np.allclose(p1, np.array([0.2, 0.8], dtype=np.float32))
    assert np.allclose(p2, np.array([0.4, 0.6], dtype=np.float32))
