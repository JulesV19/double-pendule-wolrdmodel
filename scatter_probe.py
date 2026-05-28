"""
Scatter plots : valeurs réelles vs prédites pour θ et ω,
avec probe linéaire et probe MLP côte à côte.

Grille 2×2 :
  colonnes = Linéaire | MLP
  lignes   = θ        | ω

Usage:
  python3 scatter_probe.py --checkpoint checkpoints/lewm_best.pt
  python3 scatter_probe.py --checkpoint checkpoints/lewm_best.pt --save visuals/scatter.png
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
        embed_dim=args.get("embed_dim", 128),
        hidden_dim=args.get("hidden_dim", 512),
        lam=args.get("lam", 0.5),
        n_proj=args.get("n_proj", 512),
        ema_momentum=args.get("ema_momentum", 0.996),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Checkpoint : epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', float('nan')):.5f}")
    return model


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
        return head(Zt).cpu().numpy()


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-8))


def scatter_panel(ax, true, pred, label, color):
    ax.set_facecolor(DARK)
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    ax.scatter(true, pred, s=3, alpha=0.25, color=color, rasterized=True)

    lo = min(true.min(), pred.min())
    hi = max(true.max(), pred.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            color="#ff8a65", lw=1.2, ls="--")

    score = r2(pred, true)
    ax.set_title(f"{label}   R² = {score:.4f}", color="white", fontsize=10)
    ax.set_xlabel("Valeur réelle", color="white", fontsize=8)
    ax.set_ylabel("Valeur prédite", color="white", fontsize=8)
    return score


def main(args):
    device = get_device()
    print(f"Device : {device}")

    model = load_model(args.checkpoint, device)

    ds    = PendulumSeqDataset(args.dataset_dir)
    n_val = max(1, int(len(ds) * 0.15))
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False, num_workers=0)

    print("Encodage des embeddings…")
    Z_tr, S_tr = collect(model, train_loader, device)
    Z_va, S_va = collect(model, val_loader,   device)

    D = Z_tr.shape[1]
    Zt = torch.from_numpy(Z_tr).float().to(device)
    St = torch.from_numpy(S_tr).float().to(device)
    Zv = torch.from_numpy(Z_va).float().to(device)

    print(f"Train : {len(Z_tr):,} points  |  Val : {len(Z_va):,} points  |  D={D}")

    # ── Probe linéaire ──────────────────────────────────────────────────────────
    print("Entraînement probe linéaire…")
    lin_head  = nn.Linear(D, 2).to(device)
    train_probe(lin_head, Zt, St, n_epochs=args.epochs_lin, lr=1e-3)
    with torch.no_grad():
        lin_preds = lin_head(Zv).cpu().numpy()

    # ── Probe MLP ───────────────────────────────────────────────────────────────
    print("Entraînement probe MLP…")
    mlp_head = nn.Sequential(
        nn.Linear(D, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 2),
    ).to(device)
    train_probe(mlp_head, Zt, St, n_epochs=args.epochs_mlp, lr=3e-4)
    with torch.no_grad():
        mlp_preds = mlp_head(Zv).cpu().numpy()

    # ── Figure ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), facecolor=DARK)
    fig.patch.set_facecolor(DARK)

    C_THETA = "#4fc3f7"
    C_OMEGA = "#ff8a65"

    r2_lin_theta = scatter_panel(axes[0, 0], S_va[:, 0], lin_preds[:, 0], "θ  —  Linéaire", C_THETA)
    r2_mlp_theta = scatter_panel(axes[0, 1], S_va[:, 0], mlp_preds[:, 0], "θ  —  MLP",      C_THETA)
    r2_lin_omega = scatter_panel(axes[1, 0], S_va[:, 1], lin_preds[:, 1], "ω  —  Linéaire", C_OMEGA)
    r2_mlp_omega = scatter_panel(axes[1, 1], S_va[:, 1], mlp_preds[:, 1], "ω  —  MLP",      C_OMEGA)

    fig.suptitle("Scatter probe : espace latent → état physique", color="white", fontsize=12, y=1.01)

    col_labels = ["Linéaire", "MLP"]
    for ax, label in zip(axes[0], col_labels):
        ax.set_title(f"θ  —  {label}   R² = {r2_lin_theta if label=='Linéaire' else r2_mlp_theta:.4f}",
                     color="white", fontsize=10)
    for ax, label in zip(axes[1], col_labels):
        ax.set_title(f"ω  —  {label}   R² = {r2_lin_omega if label=='Linéaire' else r2_mlp_omega:.4f}",
                     color="white", fontsize=10)

    print(f"\n{'':12} {'Linéaire':>10}  {'MLP':>10}")
    print(f"  R²(θ)      {r2_lin_theta:>10.4f}  {r2_mlp_theta:>10.4f}")
    print(f"  R²(ω)      {r2_lin_omega:>10.4f}  {r2_mlp_omega:>10.4f}")
    print(f"  R² global  {(r2_lin_theta+r2_lin_omega)/2:>10.4f}  {(r2_mlp_theta+r2_mlp_omega)/2:>10.4f}")

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight", facecolor=DARK)
        print(f"\nSauvegardé : {args.save}")
    else:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="checkpoints/lewm_best.pt")
    parser.add_argument("--dataset-dir", default="dataset/pendulum")
    parser.add_argument("--epochs-lin",  type=int, default=100)
    parser.add_argument("--epochs-mlp",  type=int, default=400)
    parser.add_argument("--save",        default=None)
    main(parser.parse_args())
