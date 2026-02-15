from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut


def save_confusion(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, ax=ax, cmap="Blues")
    ax.set_title("Confusion matrix")
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LSE TCN model")
    parser.add_argument("--model", default="outputs/best_model.pt")
    parser.add_argument("--dataset", default="data/processed/social50_dataset.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--loso", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from lse_tcn.models.tcn import LSETCN

    class SkeletonDataset(Dataset):
        def __init__(self, samples, window_size: int):
            self.samples = [s for s in samples if s["label"] >= 0]
            self.window_size = window_size

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            s = self.samples[i]
            x = s["frames"]
            t, f = x.shape
            out = np.zeros((self.window_size, f), dtype=np.float32)
            out[: min(t, self.window_size)] = x[: self.window_size]
            if t < self.window_size and t > 0:
                out[t:] = x[t - 1]
            return torch.from_numpy(out).float(), torch.tensor(int(s["label"]), dtype=torch.long)

    def eval_split(model, ds):
        model.eval()
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        ys, ps = [], []
        with torch.no_grad():
            for x, y in loader:
                pred = model(x.to(args.device)).argmax(1).cpu().numpy()
                ys.extend(y.numpy().tolist())
                ps.extend(pred.tolist())
        return np.array(ys), np.array(ps)

    ckpt = torch.load(args.model, map_location=args.device)
    samples = torch.load(args.dataset)
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

    if args.loso:
        groups = np.array([s.get("signer_id", "unknown") for s in samples])
        logo = LeaveOneGroupOut()
        for i, (_, test_idx) in enumerate(logo.split(samples, groups=groups), start=1):
            y, p = eval_split(model, SkeletonDataset([samples[j] for j in test_idx], cfg["window_size"]))
            print(f"LOSO fold={i} signer={groups[test_idx[0]]} acc={accuracy_score(y, p):.4f} macroF1={f1_score(y, p, average='macro'):.4f}")
    else:
        y, p = eval_split(model, SkeletonDataset(samples, cfg["window_size"]))
        print(f"accuracy={accuracy_score(y, p):.4f} macroF1={f1_score(y, p, average='macro'):.4f}")
        save_confusion(y, p, labels, Path("outputs/confusion_matrix.png"))


if __name__ == "__main__":
    main()
