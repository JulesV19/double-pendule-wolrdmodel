"""
Entraînement du décodeur z → frame (encodeur gelé).

Usage:
  python3 train_decoder.py --checkpoint checkpoints/lewm_best.pt
  python3 train_decoder.py --checkpoint checkpoints/lewm_best.pt --epochs 30 --lr 3e-4
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from models.lewm    import LeWorldModel
from models.decoder import Decoder
from dataset        import make_seq_dataloaders


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_encoder(path: str, device) -> LeWorldModel:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    args  = ckpt.get("args", {})
    model = LeWorldModel(
        embed_dim    = args.get("embed_dim",    128),
        hidden_dim   = args.get("hidden_dim",   512),
        lam          = args.get("lam",          0.5),
        n_proj       = args.get("n_proj",       512),
        ema_momentum = args.get("ema_momentum", 0.996),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Encodeur chargé (epoch={ckpt.get('epoch','?')})")
    return model


@torch.no_grad()
def evaluate(encoder, decoder, loader, device):
    decoder.eval()
    total = 0.0
    for frames, _ in loader:
        frames = frames.to(device)               # (B, T, 3, 64, 64)
        B, T, C, H, W = frames.shape
        z      = encoder.encode(frames)          # (B, T, D)
        recon  = decoder(z)                      # (B, T, 3, 64, 64)
        total += nn.functional.mse_loss(recon, frames).item()
    return total / len(loader)


def train(args):
    device = get_device()
    print(f"Device : {device}")

    train_loader, val_loader = make_seq_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    encoder = load_encoder(args.checkpoint, device)
    embed_dim = encoder.embed_dim

    decoder   = Decoder(embed_dim=embed_dim).to(device)
    n_params  = sum(p.numel() for p in decoder.parameters())
    print(f"Décodeur : {n_params:,} paramètres")

    optimizer = optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        t0   = time.time()
        total = 0.0

        for frames, _ in train_loader:
            frames = frames.to(device)
            B, T, C, H, W = frames.shape

            with torch.no_grad():
                z = encoder.encode(frames)       # (B, T, D)

            recon = decoder(z)                   # (B, T, 3, 64, 64)
            loss  = nn.functional.mse_loss(recon, frames)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        scheduler.step()
        train_loss = total / len(train_loader)
        val_loss   = evaluate(encoder, decoder, val_loader, device)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            torch.save({
                "epoch":     epoch,
                "decoder":   decoder.state_dict(),
                "val_loss":  best_val,
                "embed_dim": embed_dim,
            }, ckpt_dir / "decoder_best.pt")

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  train={train_loss:.4f}"
            f"  val={val_loss:.4f}"
            f"  {elapsed:.1f}s"
            + ("  <-- best" if improved else "")
        )

    print(f"\nMeilleur val MSE : {best_val:.4f}  →  {ckpt_dir}/decoder_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  required=True,
                        help="checkpoint LeWorldModel (encodeur)")
    parser.add_argument("--dataset-dir", default="dataset/double_pendulum")
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int,   default=4)
    parser.add_argument("--ckpt-dir",    default="checkpoints")
    train(parser.parse_args())
