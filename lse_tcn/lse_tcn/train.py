from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSE TCN")
    parser.add_argument("--config", default="configs/social50.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from pathlib import Path
    from typing import Any

    import numpy as np
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from tqdm import tqdm

    from lse_tcn.data.adapters_sign4all import Sign4AllAdapter
    from lse_tcn.data.adapters_swl_lse import SWLLSEAdapter
    from lse_tcn.data.augment import augment_sequence
    from lse_tcn.models.tcn import LSETCN
    from lse_tcn.utils.config import labels_to_id, load_config

    class SkeletonDataset(Dataset):
        def __init__(self, samples: list[dict[str, Any]], window_size: int, training: bool = False) -> None:
            self.samples = [s for s in samples if s["label"] >= 0]
            self.window_size = window_size
            self.training = training

        def __len__(self) -> int:
            return len(self.samples)

        def _fit_window(self, frames: np.ndarray) -> np.ndarray:
            t, f = frames.shape
            if t >= self.window_size:
                start = 0 if not self.training else np.random.randint(0, t - self.window_size + 1)
                return frames[start : start + self.window_size]
            out = np.zeros((self.window_size, f), dtype=np.float32)
            out[:t] = frames
            if t > 0:
                out[t:] = frames[-1]
            return out

        def __getitem__(self, idx: int):
            s = self.samples[idx]
            x = self._fit_window(s["frames"])
            if self.training:
                x = augment_sequence(x)
            y = int(s["label"])
            return torch.from_numpy(x).float(), torch.tensor(y, dtype=torch.long)

    def load_all_samples(cfg: dict[str, Any]) -> list[dict[str, Any]]:
        selected = set(cfg["labels"])
        label_map = labels_to_id(cfg["labels"])
        samples: list[dict[str, Any]] = []
        data_cfg = cfg["data"]
        if data_cfg["swl_lse"]["enabled"]:
            swl = SWLLSEAdapter(data_cfg["swl_lse"]["path"], data_cfg["swl_lse"].get("label_map_path"))
            for s in swl.load_samples(selected):
                s["label"] = label_map.get(s["label_name"], -1)
                samples.append(s)
        if data_cfg["sign4all"]["enabled"]:
            s4a = Sign4AllAdapter(data_cfg["sign4all"]["path"], data_cfg["sign4all"].get("label_map_path"))
            for s in s4a.load_samples(selected):
                s["label"] = label_map.get(s["label_name"], -1)
                samples.append(s)
        return samples

    def make_loader(dataset: SkeletonDataset, batch_size: int, balanced: bool) -> DataLoader:
        if not balanced or len(dataset) == 0:
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)
        labels = [int(s["label"]) for s in dataset.samples]
        bincount = np.bincount(labels)
        weights = [1.0 / max(bincount[y], 1) for y in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    def run_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer | None, device: str):
        crit = nn.CrossEntropyLoss()
        train = opt is not None
        model.train(train)
        losses, correct, total = 0.0, 0, 0
        for x, y in tqdm(loader, leave=False):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            losses += float(loss.item()) * x.size(0)
            pred = logits.argmax(1)
            correct += int((pred == y).sum().item()); total += x.size(0)
        return losses / max(total, 1), correct / max(total, 1)

    cfg = load_config(args.config).raw
    samples = load_all_samples(cfg)
    if not samples:
        raise RuntimeError("No samples found. Prepare datasets or disable missing adapters.")
    cache_path = Path(cfg["data"]["cache_path"]); cache_path.parent.mkdir(parents=True, exist_ok=True); torch.save(samples, cache_path)
    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=cfg["seed"])
    window = int(cfg["window_size"])
    train_ds = SkeletonDataset(train_samples, window_size=window, training=True)
    val_ds = SkeletonDataset(val_samples, window_size=window, training=False)
    batch = args.batch or cfg["train"]["batch_size"]
    train_loader = make_loader(train_ds, batch, cfg["train"].get("class_balanced_sampling", True))
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)
    model = LSETCN(input_dim=cfg["model"]["input_dim"], num_classes=len(cfg["labels"]), channels=cfg["model"]["channels"], dilations=cfg["model"]["dilations"], kernel_size=cfg["model"]["kernel_size"], dropout=cfg["model"]["dropout"]).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr or cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    epochs = args.epochs or cfg["train"]["epochs"]
    best_acc = -1.0
    out_dir = Path(cfg.get("output_dir", "outputs")); out_dir.mkdir(exist_ok=True)
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, opt, args.device)
        va_loss, va_acc = run_epoch(model, val_loader, None, args.device)
        print(f"epoch={epoch} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(), "labels": cfg["labels"], "config": cfg}, out_dir / "best_model.pt")


if __name__ == "__main__":
    main()
