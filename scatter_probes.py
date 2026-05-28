"""
Scatter plots θ et ω : probe linéaire vs MLP.

Usage:
  python3 scatter_probes.py --checkpoint checkpoints/lewm_best.pt
  python3 scatter_probes.py --checkpoint checkpoints/lewm_best.pt --save visuals/scatter_probes.png
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from models.lewm import LeWorldModel
from dataset import PendulumSeqDataset

DARK = "#111"


def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_model(path, device):
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    args  = ckpt.get("args", {})
    model = LeWorldModel(
        embed_dim    = args.get("embed_dim",    128),
        hidden_dim   = args.get("hidden_dim",   512),
        lam          = args.get("lam",          0.5),
        ema_momentum = args.get("ema_momentum", 0.996),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Checkpoint chargé  epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', float('nan')):.5f}")
    return model


def make_loaders(dataset_dir, val_split=0.1, seed=42):
    ds    = PendulumSeqDataset(dataset_dir)
    n_val = int(len(ds) * val_split)
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return (DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0))


@torch.no_grad()
def collect(model, loader, device):
    Zs, Ss = [], []
    for frames, states in loader:
        z = model.encode(frames.to(device))   # (B, T, D)
        Zs.append(z.cpu().numpy().reshape(-1, z.shape[-1]))
        Ss.append(states.numpy().reshape(-1, states.shape[-1]))
    return np.concatenate(Zs), np.concatenate(Ss)


def train_probe(head, Zt, St, n_epochs, lr):
    opt = optim.Adam(head.parameters(), lr=lr)
    for _ in range(n_epochs):
        head.train()
        opt.zero_grad()
        F.mse_loss(head(Zt), St).backward()
        opt.step()
    head.eval()
    with torch.no_grad():
        return head(Zt).cpu().numpy()   # train preds (unused)


def r2(y_pred, y_true):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-8))


def run(args):
    device = get_device()
    model  = load_model(args.checkpoint, device)
    train_loader, val_loader = make_loaders(args.dataset_dir)

    print("Collecte des embeddings…")
    Z_tr, S_tr = collect(model, train_loader, device)
    Z_va, S_va = collect(model, val_loader,   device)

    D = Z_tr.shape[1]
    Zt = torch.from_numpy(Z_tr).float().to(device)
    St = torch.from_numpy(S_tr).float().to(device)
    Zv = torch.from_numpy(Z_va).float().to(device)

    # Probe linéaire
    print("Entraînement probe linéaire…")
    lin = nn.Linear(D, 2).to(device)
    train_probe(lin, Zt, St, n_epochs=args.probe_epochs, lr=1e-3)
    with torch.no_grad():
        lin_preds = lin(Zv).cpu().numpy()

    # Probe MLP
    print("Entraînement probe MLP…")
    mlp = nn.Sequential(
        nn.Linear(D, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 2),
    ).to(device)
    train_probe(mlp, Zt, St, n_epochs=args.probe_epochs * 4, lr=3e-4)
    with torch.no_grad():
        mlp_preds = mlp(Zv).cpu().numpy()

    # ── Figure : 2 lignes × 2 colonnes ────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), facecolor=DARK)
    fig.suptitle("Scatter probes — Linéaire vs MLP", color="white", fontsize=13, y=0.99)

    configs = [
        # (ax,         state_idx, state_name, preds,     color,     probe_label)
        (axes[0, 0],   0,         "θ",        lin_preds, "#4fc3f7", "Linéaire"),
        (axes[0, 1],   0,         "θ",        mlp_preds, "#ff8a65", "MLP"),
        (axes[1, 0],   1,         "ω",        lin_preds, "#4fc3f7", "Linéaire"),
        (axes[1, 1],   1,         "ω",        mlp_preds, "#ff8a65", "MLP"),
    ]

    for ax, idx, name, preds, color, probe_label in configs:
        ax.set_facecolor(DARK)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

        y_true = S_va[:, idx]
        y_pred = preds[:, idx]
        score  = r2(y_pred, y_true)

        ax.scatter(y_true, y_pred, s=3, alpha=0.25, color=color, rasterized=True)

        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], color="white", lw=1.0, ls="--", alpha=0.5)

        ax.set_xlabel(f"{name} réel",  color="white", fontsize=9)
        ax.set_ylabel(f"{name} prédit", color="white", fontsize=9)
        ax.set_title(f"{name}  —  {probe_label}   R²={score:.3f}",
                     color=color, fontsize=10)

        print(f"  R²({name}, {probe_label:8s}) = {score:.4f}")

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight", facecolor=DARK)
        print(f"\nFigure sauvegardée : {args.save}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--dataset-dir",  default="dataset/double_pendulum")
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--save",         default=None)
    run(parser.parse_args())
