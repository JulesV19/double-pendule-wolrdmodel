"""
Entraînement LeWorldModel (JEPA + SIGReg).

Pas d'encodeur cible EMA, pas de VICReg, pas de masquage complexe.
Un seul hyperparamètre effectif : λ (poids SIGReg).

Usage:
  python3 train_lewm.py
  python3 train_lewm.py --lam 0.1 --embed-dim 128 --epochs 100
  python3 train_lewm.py --checkpoint checkpoints/lewm_best.pt   # resume
"""

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from models.lewm import LeWorldModel
from dataset import make_seq_dataloaders


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():        return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: LeWorldModel, loader, device) -> dict:
    model.eval()
    sums = {"loss": 0.0, "pred_loss": 0.0, "sigreg": 0.0}
    for frames, _ in loader:
        m = model(frames.to(device, non_blocking=True))
        for k in sums:
            sums[k] += m[k].item()
    n = len(loader)
    return {k: v / n for k, v in sums.items()}


# ── Training ────────────────────────────────────────────────────────────────────

def train(args):
    device = get_device()
    print(f"Device : {device}")

    train_loader, val_loader = make_seq_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"Train : {len(train_loader)} batches  |  Val : {len(val_loader)} batches")

    model = LeWorldModel(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        lam=args.lam,
        n_proj=args.n_proj,
        ema_momentum=args.ema_momentum,
        mask_ratio=args.mask_ratio,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres : {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    # Warmup linéaire 5 epochs → cosine decay
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        t = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * t)).item())

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 1
    if args.checkpoint:
        # torch.load does not accept a `weights_only` kwarg; use map_location only
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Reprise depuis epoch {ckpt['epoch']}")

    ckpt_dir = Path(args.ckpt_dir)
    vis_dir  = Path(args.vis_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    history   = {"train": [], "val": [], "pred": [], "sigreg": []}
    best_val  = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0   = time.time()
        sums = {"loss": 0.0, "pred_loss": 0.0, "sigreg": 0.0}

        for frames, _ in train_loader:
            frames = frames.to(device, non_blocking=True)
            optimizer.zero_grad()
            m = model(frames)
            m["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            model.update_target()
            for k in sums:
                sums[k] += m[k].item()

        scheduler.step()
        n = len(train_loader)
        train_loss = sums["loss"] / n
        val_m      = evaluate(model, val_loader, device)

        history["train"].append(train_loss)
        history["val"].append(val_m["loss"])
        history["pred"].append(sums["pred_loss"] / n)
        history["sigreg"].append(sums["sigreg"] / n)

        improved = val_m["loss"] < best_val
        if improved:
            best_val = val_m["loss"]
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss":  best_val,
                "args":      vars(args),
            }, ckpt_dir / "lewm_best.pt")

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  loss={train_loss:.4f}"
            f"  pred={sums['pred_loss']/n:.4f}"
            f"  sig={sums['sigreg']/n:.4f}"
            f"  val={val_m['loss']:.4f}"
            f"  lr={lr_now:.2e}"
            f"  {elapsed:.1f}s"
            + ("  <-- best" if improved else "")
        )

    _save_plot(history, vis_dir / "lewm_loss.png")
    print(f"\nCheckpoints → {ckpt_dir}/")
    print(f"Visuals     → {vis_dir}/")


# ── Plot ───────────────────────────────────────────────────────────────────────

def _save_plot(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#111")
    epochs = range(1, len(history["train"]) + 1)

    ax = axes[0]
    ax.set_facecolor("#111")
    ax.plot(epochs, history["train"], color="#4fc3f7", label="train")
    ax.plot(epochs, history["val"],   color="#ff8a65", label="val")
    ax.set_title("Loss totale", color="white")
    ax.set_xlabel("epoch", color="white")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    ax = axes[1]
    ax.set_facecolor("#111")
    ax.plot(epochs, history["pred"],   color="#a5d6a7", label="pred MSE")
    ax.plot(epochs, history["sigreg"], color="#ce93d8", label="SIGReg")
    ax.set_title("Décomposition", color="white")
    ax.set_xlabel("epoch", color="white")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor="#111")
    plt.close()


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir",  default="dataset/double_pendulum")
    parser.add_argument("--embed-dim",    type=int,   default=128)
    parser.add_argument("--hidden-dim",   type=int,   default=512)
    parser.add_argument("--n-heads",      type=int,   default=4)
    parser.add_argument("--n-layers",     type=int,   default=4)
    parser.add_argument("--lam",          type=float, default=0.5,
                        help="poids SIGReg")
    parser.add_argument("--ema-momentum", type=float, default=0.99,
                        help="momentum EMA du target encoder (τ)")
    parser.add_argument("--mask-ratio",   type=float, default=0.4,
                        help="fraction des frames masquées pour la prédiction (style V-JEPA)")
    parser.add_argument("--n-proj",       type=int,   default=512,
                        help="projections SIGReg (robuste à ce choix)")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--ckpt-dir",     default="checkpoints")
    parser.add_argument("--vis-dir",      default="visuals")
    parser.add_argument("--checkpoint",   default=None)
    args = parser.parse_args()
    train(args)
