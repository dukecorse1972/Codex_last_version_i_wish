from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from lse_tcn.data.preprocess import extract_frame_features, movement_magnitude, normalize_frame


@dataclass
class StreamingState:
    window_size: int
    infer_stride: int
    smooth_window: int

    def __post_init__(self) -> None:
        self.buffer: deque[np.ndarray] = deque(maxlen=self.window_size)
        self.prob_history: deque[np.ndarray] = deque(maxlen=self.smooth_window)
        self.frame_counter = 0

    def push(self, frame_feature: np.ndarray) -> None:
        self.buffer.append(frame_feature)
        self.frame_counter += 1

    def ready(self) -> bool:
        return len(self.buffer) == self.window_size and self.frame_counter % self.infer_stride == 0

    def get_window(self) -> np.ndarray:
        return np.stack(self.buffer, axis=0)

    def smooth(self, probs: np.ndarray) -> np.ndarray:
        self.prob_history.append(probs)
        return np.mean(np.stack(self.prob_history, axis=0), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime LSE demo")
    parser.add_argument("--model", default="outputs/best_model.pt")
    parser.add_argument("--webcam-index", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--infer-stride", type=int, default=5)
    parser.add_argument("--movement-threshold", type=float, default=0.015)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--rotate-align", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import cv2
    import mediapipe as mp
    import torch
    from lse_tcn.models.tcn import LSETCN

    ckpt = torch.load(args.model, map_location=args.device)
    labels = ckpt["labels"]
    cfg = ckpt["config"]
    model = LSETCN(
        input_dim=cfg["model"]["input_dim"],
        num_classes=len(labels),
        channels=cfg["model"]["channels"],
        dilations=cfg["model"]["dilations"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=cfg["model"]["dropout"],
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    state = StreamingState(args.window_size, args.infer_stride, args.smoothing_window)
    cap = cv2.VideoCapture(args.webcam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    mp_holistic = mp.solutions.holistic
    pred_text = "NO_ENTIENDO"
    conf = 0.0
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            feat = normalize_frame(extract_frame_features(results), rotate_align=args.rotate_align)
            state.push(feat)

            if state.ready():
                window = state.get_window()
                if movement_magnitude(window) > args.movement_threshold:
                    x = torch.from_numpy(window[None, ...]).float().to(args.device)
                    with torch.no_grad():
                        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
                    smooth = state.smooth(probs)
                    idx = int(np.argmax(smooth))
                    conf = float(smooth[idx])
                    pred_text = labels[idx] if conf >= args.confidence_threshold else "NO_ENTIENDO"
                else:
                    pred_text, conf = "SILENCIO", 0.0

            latency_ms = (time.perf_counter() - t0) * 1000.0
            fps = 1000.0 / max(latency_ms, 1e-3)
            overlay = f"{pred_text} | conf={conf:.2f} | FPS={fps:.1f} | lat={latency_ms:.1f}ms"
            cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("LSE realtime", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
