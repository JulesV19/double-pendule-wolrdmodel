"""
Entraînement AutoEncoder baseline.

Supervision dans l'espace pixel (reconstruction MSE) — à comparer avec
LeWorldModel (supervision dans l'espace latent via target encoder EMA).

Usage:
  python3 train_ae.py
  python3 train_ae.py --embed-dim 128 --rollout-k 5 --epochs 50
  python3 train_ae.py --checkpoint checkpoints/ae_best.pt   # resume
"""

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from models.ae import AutoEncoder
from dataset import make_seq_dataloaders


# ── Device ─────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── Mémoire GPU ────────────────────────────────────────────────────────────────

def peak_memory_mb(device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return 0.0


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: AutoEncoder, loader, device) -> dict:
    model.eval()
    total = 0.0
    for frames, _ in loader:
        m = model(frames.to(device, non_blocking=True))
        total += m["loss"].item()
    return {"loss": total / len(loader), "recon_loss": total / len(loader)}


# ── Training ────────────────────────────────────────────────────────────────────

def train(args):
    device = get_device()
    print(f"Device : {device}")

    train_loader, val_loader = make_seq_dataloaders(
        dataset_dir=args.dataset_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seq_len=args.seq_len,
    )
    print(f"Train : {len(train_loader)} batches  |  Val : {len(val_loader)} batches")

    model = AutoEncoder(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        rollout_k=args.rollout_k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres : {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return (epoch + 1) / warmup
        t = (epoch - warmup) / max(1, args.epochs - warmup)
        return 0.5 * (1 + torch.cos(torch.tensor(3.14159 * t)).item())

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 1
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Reprise depuis epoch {ckpt['epoch']}")

    ckpt_dir = Path(args.ckpt_dir)
    vis_dir  = Path(args.vis_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history  = {"train": [], "val": [], "epoch_time": []}
    best_val = float("inf")
    total_train_time = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0    = time.time()
        total = 0.0

        for frames, _ in train_loader:
            frames = frames.to(device, non_blocking=True)
            optimizer.zero_grad()
            m = model(frames, var_lambda=args.var_lambda)
            m["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total += m["loss"].item()

        scheduler.step()
        elapsed = time.time() - t0
        total_train_time += elapsed

        n          = len(train_loader)
        train_loss = total / n
        val_m      = evaluate(model, val_loader, device)

        history["train"].append(train_loss)
        history["val"].append(val_m["loss"])
        history["epoch_time"].append(elapsed)

        improved = val_m["loss"] < best_val
        if improved:
            best_val = val_m["loss"]
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss":  best_val,
                "args":      vars(args),
            }, ckpt_dir / "ae_best.pt")

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  recon={train_loss:.5f}"
            f"  val={val_m['loss']:.5f}"
            f"  lr={lr_now:.2e}"
            f"  {elapsed:.1f}s"
            + ("  <-- best" if improved else "")
        )
        if epoch == 1 or epoch % 5 == 0:
            with torch.no_grad():
                m_dbg = model(next(iter(val_loader))[0].to(device), var_lambda=0.0)
                print(f"         recon_only={m_dbg['recon_loss'].item():.5f}"
                      f"  var_loss={m_dbg['var_loss'].item():.4f}")

    mem_mb = peak_memory_mb(device)
    avg_epoch_time = total_train_time / (args.epochs - start_epoch + 1)
    print(f"\nTemps moyen / epoch : {avg_epoch_time:.1f}s")
    if mem_mb:
        print(f"Pic mémoire GPU     : {mem_mb:.0f} MB")

    _save_plot(history, vis_dir / "ae_loss.png")
    print(f"Checkpoints → {ckpt_dir}/")
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
    ax.set_title("Reconstruction MSE", color="white")
    ax.set_xlabel("epoch", color="white")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    ax = axes[1]
    ax.set_facecolor("#111")
    ax.plot(epochs, history["epoch_time"], color="#a5d6a7", label="temps/epoch (s)")
    ax.set_title("Temps par epoch", color="white")
    ax.set_xlabel("epoch", color="white")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor="#111")
    plt.close()


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement AutoEncoder baseline")
    parser.add_argument("--dataset-dir",  default="dataset/pendulum")
    parser.add_argument("--seq-len",      type=int,   default=100)
    parser.add_argument("--embed-dim",    type=int,   default=128)
    parser.add_argument("--hidden-dim",   type=int,   default=512)
    parser.add_argument("--rollout-k",    type=int,   default=5,
                        help="nombre de steps de rollout pour la recon loss")
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch-size",   type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--ckpt-dir",     default="checkpoints")
    parser.add_argument("--vis-dir",      default="visuals")
    parser.add_argument("--checkpoint",   default=None)
    parser.add_argument("--var-lambda",   type=float, default=0.1,
                        help="poids régularisation variance anti-collapse")
    args = parser.parse_args()
    train(args)
