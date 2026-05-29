"""
Entraînement RSSM (Recurrent State Space Model) baseline.

Supervision pixel (reconstruction MSE + KL prior/posterior) — à comparer avec
LeWorldModel (supervision dans l'espace latent via cosine loss + EMA).

Usage:
  python3 train_rssm.py
  python3 train_rssm.py --feat-dim 128 --h-dim 200 --s-dim 32 --epochs 100
  python3 train_rssm.py --checkpoint checkpoints/rssm_best.pt   # resume
"""

import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from models.rssm import RSSM
from dataset import make_seq_dataloaders


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def peak_memory_mb(device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    return 0.0


@torch.no_grad()
def evaluate(model: RSSM, loader, device, kl_scale, pixel_weight) -> dict:
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    for frames, _ in loader:
        m = model(frames.to(device, non_blocking=True),
                  kl_scale=kl_scale, pixel_weight=pixel_weight)
        total_loss  += m["loss"].item()
        total_recon += m["recon_loss"].item()
        total_kl    += m["kl_loss"].item()
    n = len(loader)
    return {"loss": total_loss / n, "recon_loss": total_recon / n, "kl_loss": total_kl / n}


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

    model = RSSM(
        feat_dim=args.feat_dim,
        h_dim=args.h_dim,
        s_dim=args.s_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paramètres : {n_params:,}")
    print(f"Latent dim : h={args.h_dim}  s={args.s_dim}  total={args.h_dim + args.s_dim}")

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
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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

    history  = {"train": [], "val": [], "kl_loss": [], "epoch_time": []}
    best_val = float("inf")
    total_train_time = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss = total_kl = 0.0

        for frames, _ in train_loader:
            frames = frames.to(device, non_blocking=True)
            optimizer.zero_grad()
            m = model(frames,
                      kl_scale=args.kl_scale,
                      pixel_weight=args.pixel_weight,
                      free_nats=args.free_nats)
            m["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
            optimizer.step()
            total_loss += m["loss"].item()
            total_kl   += m["kl_loss"].item()

        scheduler.step()
        elapsed = time.time() - t0
        total_train_time += elapsed

        n          = len(train_loader)
        train_loss = total_loss / n
        kl_loss    = total_kl / n
        val_m      = evaluate(model, val_loader, device,
                              args.kl_scale, args.pixel_weight)

        history["train"].append(train_loss)
        history["val"].append(val_m["loss"])
        history["kl_loss"].append(kl_loss)
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
            }, ckpt_dir / "rssm_best.pt")

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{args.epochs}"
            f"  loss={train_loss:.5f}"
            f"  kl={kl_loss:.3f}"
            f"  val={val_m['loss']:.5f}"
            f"  lr={lr_now:.2e}"
            f"  {elapsed:.1f}s"
            + ("  <-- best" if improved else "")
        )
        if epoch == 1 or epoch % 5 == 0:
            print(f"         recon={val_m['recon_loss']:.5f}"
                  f"  kl_val={val_m['kl_loss']:.3f}")

    mem_mb = peak_memory_mb(device)
    avg_t  = total_train_time / (args.epochs - start_epoch + 1)
    print(f"\nTemps moyen / epoch : {avg_t:.1f}s")
    if mem_mb:
        print(f"Pic mémoire GPU     : {mem_mb:.0f} MB")

    _save_plot(history, vis_dir / "rssm_loss.png")
    print(f"Checkpoints → {ckpt_dir}/")
    print(f"Visuals     → {vis_dir}/")


def _save_plot(history, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#111")
    epochs = range(1, len(history["train"]) + 1)

    panels = [
        (axes[0], "Loss totale (recon + kl_scale*KL)", history["train"], history["val"],
         "#4fc3f7", "#ff8a65", "train", "val"),
        (axes[1], "KL loss (train)", history["kl_loss"], None,
         "#a5d6a7", None, "kl_loss", None),
        (axes[2], "Temps par epoch (s)", history["epoch_time"], None,
         "#ce93d8", None, "temps/epoch", None),
    ]

    for ax, title, data1, data2, c1, c2, l1, l2 in panels:
        ax.set_facecolor("#111")
        ax.plot(epochs, data1, color=c1, label=l1)
        if data2 is not None:
            ax.plot(epochs, data2, color=c2, label=l2)
        ax.set_title(title, color="white")
        ax.set_xlabel("epoch", color="white")
        ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches="tight", facecolor="#111")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement RSSM baseline")
    parser.add_argument("--dataset-dir",  default="dataset/pendulum")
    parser.add_argument("--seq-len",      type=int,   default=50,
                        help="longueur de séquence (GRU séquentiel, plus court que AE)")
    parser.add_argument("--feat-dim",     type=int,   default=128,
                        help="sortie encodeur CNN (= embed_dim AE pour comparaison équitable)")
    parser.add_argument("--h-dim",        type=int,   default=200,
                        help="état déterministe GRU")
    parser.add_argument("--s-dim",        type=int,   default=32,
                        help="état stochastique")
    parser.add_argument("--hidden-dim",   type=int,   default=256,
                        help="taille MLP prior/posterior")
    parser.add_argument("--kl-scale",     type=float, default=1.0,
                        help="poids du terme KL dans la loss")
    parser.add_argument("--free-nats",    type=float, default=3.0,
                        help="plancher KL sur le scalaire final (PlaNet default=3.0)")
    parser.add_argument("--pixel-weight", type=float, default=10.0,
                        help="sur-pondération pixels brillants dans wmse")
    parser.add_argument("--epochs",       type=int,   default=100)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=6e-4,
                        help="lr AdamW (DreamerV2 utilise 6e-4)")
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--ckpt-dir",     default="checkpoints")
    parser.add_argument("--vis-dir",      default="visuals")
    parser.add_argument("--checkpoint",   default=None)
    args = parser.parse_args()
    train(args)
