from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LSE TCN to ONNX")
    parser.add_argument("--model", default="outputs/best_model.pt")
    parser.add_argument("--output", default="outputs/lse_tcn.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch
    from lse_tcn.models.tcn import LSETCN

    ckpt = torch.load(args.model, map_location=args.device)
    cfg = ckpt["config"]
    model = LSETCN(
        input_dim=cfg["model"]["input_dim"],
        num_classes=len(ckpt["labels"]),
        channels=cfg["model"]["channels"],
        dilations=cfg["model"]["dilations"],
        kernel_size=cfg["model"]["kernel_size"],
        dropout=cfg["model"]["dropout"],
    ).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.randn(1, cfg["window_size"], cfg["model"]["input_dim"], device=args.device)
    torch.onnx.export(
        model,
        dummy,
        args.output,
        input_names=["skeleton"],
        output_names=["logits"],
        dynamic_axes={"skeleton": {0: "batch", 1: "time"}, "logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"Exported ONNX to {args.output}")


if __name__ == "__main__":
    main()
