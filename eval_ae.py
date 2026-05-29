"""
Évaluation AutoEncoder baseline.

Métriques adaptées à l'AE — comparables à eval_lewm.py là où c'est pertinent :
  1. Linear probe      — R² d'une régression linéaire z → (θ, ω)
  2. Uniformité &      — Détecte le collapse, cohérence temporelle
     Alignement
  3. Horizon pixel     — MSE pixel à t+1, t+2, t+5, t+10  (métrique naturelle AE)
                         (vs cosine similarity latente pour JEPA)
  4. Reconstruction    — MSE pixel moyen sur la validation

Usage:
  python3 eval_ae.py --checkpoint checkpoints/ae_best.pt
  python3 eval_ae.py --checkpoint checkpoints/ae_best.pt --save visuals/eval_ae.png
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, random_split

from models.ae import AutoEncoder
from dataset import PendulumSeqDataset


DARK        = "#111"
STATE_NAMES = ["θ", "ω"]


# ── Setup ──────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str, device) -> AutoEncoder:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    args  = ckpt.get("args", {})
    model = AutoEncoder(
        embed_dim=args.get("embed_dim", 128),
        hidden_dim=args.get("hidden_dim", 512),
        rollout_k=args.get("rollout_k", 5),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"AutoEncoder : epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.6f}")
    return model


def make_loaders(dataset_dir: str, val_split=0.1, seed=42):
    ds    = PendulumSeqDataset(dataset_dir)
    n_val = int(len(ds) * val_split)
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return (DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0))


# ── Collecte des embeddings ────────────────────────────────────────────────────

@torch.no_grad()
def collect_embeddings(model, loader, device, normalize=False):
    all_z, all_s, seqs = [], [], []
    for frames, states in loader:
        z    = model.encode(frames.to(device))
        if normalize:
            z = F.normalize(z, dim=-1)
        z_np = z.cpu().numpy()
        s_np = states.numpy()
        for b in range(len(z_np)):
            all_z.append(z_np[b])
            all_s.append(s_np[b])
            seqs.append(z_np[b])
    return np.concatenate(all_z), np.concatenate(all_s), seqs


# ── Utilitaire R² ──────────────────────────────────────────────────────────────

def compute_r2s(preds: np.ndarray, targets: np.ndarray):
    r2s = []
    for i in range(targets.shape[1]):
        ss_res = ((targets[:, i] - preds[:, i]) ** 2).sum()
        ss_tot = ((targets[:, i] - targets[:, i].mean()) ** 2).sum()
        r2s.append(float(1 - ss_res / (ss_tot + 1e-8)))
    return r2s


def _run_probe(head, Zt, St, Zv, n_epochs, lr=1e-3):
    opt = optim.Adam(head.parameters(), lr=lr)
    for _ in range(n_epochs):
        head.train(); opt.zero_grad()
        F.mse_loss(head(Zt), St).backward(); opt.step()
    head.eval()
    with torch.no_grad():
        return head(Zv).cpu().numpy()


# ── 1. Linear probe + MLP probe ────────────────────────────────────────────────

def linear_probe(model, train_loader, val_loader, device, n_epochs=50):
    print("\n── Linear probe  vs  MLP probe ──────────────────────────")
    Z_tr, S_tr, _ = collect_embeddings(model, train_loader, device)
    Z_va, S_va, _ = collect_embeddings(model, val_loader,   device)

    D        = Z_tr.shape[1]
    n_states = S_tr.shape[1]
    Zt = torch.from_numpy(Z_tr).float().to(device)
    St = torch.from_numpy(S_tr).float().to(device)
    Zv = torch.from_numpy(Z_va).float().to(device)

    lin_preds = _run_probe(nn.Linear(D, n_states).to(device), Zt, St, Zv, n_epochs)
    r2s_lin   = compute_r2s(lin_preds, S_va)
    # Clamp pour éviter des valeurs extrêmes dans le scatter si la probe diverge
    lin_preds = np.clip(lin_preds, S_va.min() - 1, S_va.max() + 1)

    mlp = nn.Sequential(
        nn.Linear(D, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, n_states),
    ).to(device)
    mlp_preds = _run_probe(mlp, Zt, St, Zv, n_epochs * 4, lr=3e-4)
    r2s_mlp   = compute_r2s(mlp_preds, S_va)

    r2_lin = float(np.mean(r2s_lin))
    r2_mlp = float(np.mean(r2s_mlp))

    print(f"  {'':6}  {'Linéaire':>10}  {'MLP':>10}")
    for name, rl, rm in zip(STATE_NAMES, r2s_lin, r2s_mlp):
        print(f"  R²({name})  {rl:>10.4f}  {rm:>10.4f}")
    print(f"  {'global':6}  {r2_lin:>10.4f}  {r2_mlp:>10.4f}")

    gap = r2_mlp - r2_lin
    if gap > 0.05:
        print(f"  → Info présente mais non-linéaire (gap = +{gap:.3f})")
    else:
        print(f"  → Peu de gain non-linéaire (gap = +{gap:.3f})")

    return r2s_lin, r2_lin, lin_preds, S_va, r2s_mlp, r2_mlp


# ── 2. Uniformité & Alignement ─────────────────────────────────────────────────

def uniformity_alignment(seqs_train, seqs_val):
    print("\n── Uniformité & Alignement ───────────────────────────────")
    Z    = np.concatenate(seqs_val)
    idx  = np.random.choice(len(Z), size=min(2000, len(Z)), replace=False)
    Zs   = Z[idx]
    d2   = np.sum((Zs[:, None] - Zs[None, :]) ** 2, axis=-1)
    mask = ~np.eye(len(Zs), dtype=bool)
    uniformity = float(np.log(np.exp(-2 * d2[mask]).mean() + 1e-8))

    align_vals = [((z[1:] - z[:-1]) ** 2).sum(axis=-1).mean() for z in seqs_train]
    alignment  = float(np.mean(align_vals))

    print(f"  Uniformité = {uniformity:.4f}  (cible : -2 à -4,  0 = collapse)")
    print(f"  Alignement = {alignment:.4f}  (cible : < 0.5)")
    return uniformity, alignment


# ── 3. Horizon de prédiction pixel ────────────────────────────────────────────
#
# Pour l'AE : la métrique naturelle est la MSE pixel à chaque horizon k.
# Le predictor a été entraîné à minimiser cette quantité — pas la cosine
# similarity dans l'espace latent (qui est la métrique de JEPA).

@torch.no_grad()
def prediction_horizon(model: AutoEncoder, val_loader, device,
                       horizons=(1, 2, 5, 10)):
    print("\n── Horizon de prédiction (MSE pixel) ────────────────────")
    results = {h: [] for h in horizons}

    for frames, _ in val_loader:
        frames = frames.to(device)
        B, T, C, H, W = frames.shape

        pairs = model._make_pairs(frames)                              # (B, T, 6, H, W)
        z_all = model.encoder(pairs.reshape(B * T, 6, H, W)).view(B, T, model.embed_dim)

        for h in horizons:
            if T <= h:
                continue
            z_roll = z_all[:, :T - h]                                 # (B, T-h, D)
            for _ in range(h):
                z_roll = model.predictor(z_roll)
            frame_pred = model.decoder(z_roll)                        # (B, T-h, 3, H, W)
            frame_tgt  = frames[:, h:]                                # (B, T-h, 3, H, W)
            mse = F.mse_loss(frame_pred, frame_tgt).item()
            results[h].append(mse)

    print(f"  {'Horizon':>8}  {'MSE pixel':>10}")
    mses = {}
    for h in horizons:
        m = float(np.mean(results[h])) if results[h] else float("nan")
        mses[h] = m
        print(f"  t+{h:>6}  {m:>10.6f}")
    return mses


# ── 4. Reconstruction MSE ──────────────────────────────────────────────────────

@torch.no_grad()
def reconstruction_quality(model: AutoEncoder, val_loader, device):
    print("\n── Reconstruction pixel MSE (k=1…rollout_k, moyenne) ────")
    total = 0.0
    n     = 0
    for frames, _ in val_loader:
        m = model(frames.to(device))
        total += m["recon_loss"].item()
        n += 1
    mse = total / n
    print(f"  MSE pixel val = {mse:.6f}")
    return mse


# ── Figure de synthèse ─────────────────────────────────────────────────────────

def save_figure(r2s, r2_global, uniformity, alignment, horizon_mses,
                recon_mse, preds, s_val, save_path, r2s_mlp=None, r2_mlp=None):
    fig = plt.figure(figsize=(15, 9), facecolor=DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    def style(ax):
        ax.set_facecolor(DARK)
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    # R² par dimension
    ax = fig.add_subplot(gs[0, 0]); style(ax)
    colors = ["#4fc3f7", "#ff8a65", "#a5d6a7", "#ce93d8"]
    x = np.arange(len(STATE_NAMES))
    w = 0.35
    ax.bar(x - w/2, r2s, width=w, color=colors, alpha=0.85,
           label=f"Linéaire R²={r2_global:.3f}")
    if r2s_mlp:
        ax.bar(x + w/2, r2s_mlp, width=w, color=colors, alpha=0.45,
               hatch="//", label=f"MLP R²={r2_mlp:.3f}")
    ax.axhline(1.0, color="#555", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(STATE_NAMES)
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("Probe R² par état", color="white", fontsize=10)
    ax.legend(fontsize=7, labelcolor="white", facecolor="#222", edgecolor="#444")

    # Scatter θ — axes bornés pour éviter explosion si probe diverge
    ax2 = fig.add_subplot(gs[0, 1]); style(ax2)
    lo = float(s_val[:, 0].min()); hi = float(s_val[:, 0].max())
    ax2.scatter(s_val[:, 0], preds[:, 0], s=4, alpha=0.3, color="#4fc3f7")
    ax2.plot([lo, hi], [lo, hi], color="#ff8a65", lw=1.2, ls="--", label="y=x")
    ax2.set_xlim(lo - 0.1, hi + 0.1)
    ax2.set_ylim(lo - 0.5, hi + 0.5)
    ax2.set_xlabel("θ réel", color="white", fontsize=9)
    ax2.set_ylabel("θ prédit (probe)", color="white", fontsize=9)
    ax2.set_title(f"Scatter θ  (R²={r2s[0]:.3f})", color="white", fontsize=10)
    ax2.legend(fontsize=8, labelcolor="white", facecolor="#222", edgecolor="#444")

    # Uniformité / Alignement
    ax3 = fig.add_subplot(gs[0, 2]); style(ax3)
    vals    = [uniformity, alignment]
    bar_col = ["#f44336" if uniformity > -1 else "#4caf50", "#4fc3f7"]
    ax3.bar(["Uniformité", "Alignement"], vals, color=bar_col, alpha=0.85)
    ax3.axhline(0, color="#555", lw=0.8)
    ax3.set_title("Uniformité & Alignement", color="white", fontsize=10)
    # ylim explicite pour que les étiquettes restent dans la figure même si vals ≈ 0
    margin = max(abs(min(vals)) * 0.3, 0.15)
    ax3.set_ylim(min(0, min(vals)) - margin, max(0, max(vals)) + margin)
    for i, v in enumerate(vals):
        offset = margin * 0.3 if v >= 0 else -margin * 0.3
        ax3.text(i, v + offset, f"{v:.3f}", ha="center", color="white", fontsize=9)

    # Horizon de prédiction pixel (courbe MSE croissante avec k)
    ax4 = fig.add_subplot(gs[1, :2]); style(ax4)
    hs   = list(horizon_mses.keys())
    mses = list(horizon_mses.values())
    ax4.plot(hs, mses, color="#4fc3f7", lw=2, marker="o", markersize=7)
    ax4.fill_between(hs, mses, alpha=0.15, color="#4fc3f7")
    ax4.axhline(0.0, color="#555", lw=0.8, ls="--", label="MSE=0 (parfait)")
    ax4.set_xlabel("Horizon (frames)", color="white", fontsize=9)
    ax4.set_ylabel("MSE pixel", color="white", fontsize=9)
    ax4.set_title("Qualité de prédiction pixel selon l'horizon", color="white", fontsize=10)
    ax4.set_xticks(hs); ax4.set_xticklabels([f"t+{h}" for h in hs])
    ax4.legend(fontsize=8, labelcolor="white", facecolor="#222", edgecolor="#444")
    for h, m in zip(hs, mses):
        ax4.annotate(f"{m:.4f}", (h, m), textcoords="offset points",
                     xytext=(0, 10), ha="center", color="white", fontsize=8)

    # Scorecard
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor("#1a1a1a"); ax5.axis("off")
    grade   = lambda r: "✓ Bon" if r > 0.8 else ("~ Moyen" if r > 0.5 else "✗ Faible")
    grade_u = lambda u: "✓ Bon" if u < -1.5 else ("~ Limite" if u < -0.5 else "✗ Collapse")
    mlp_str = f"{r2_mlp:.3f}  {grade(r2_mlp)}" if r2_mlp is not None else "—"
    lines = [
        ("SCORECARD",   "",                                          "white"),
        ("",            "",                                          "white"),
        ("R² lin. θ",   f"{r2s[0]:.3f}",                           "#a5d6a7"),
        ("R² lin. ω",   f"{r2s[1]:.3f}",                           "#a5d6a7"),
        ("R² MLP",      mlp_str,                                     "#a5d6a7"),
        ("Uniformité",  f"{uniformity:.3f}  {grade_u(uniformity)}", "#4fc3f7"),
        ("Recon. MSE",  f"{recon_mse:.5f}",                         "#ffcc80"),
        ("Pred t+1",    f"{horizon_mses.get(1, float('nan')):.5f}", "#ce93d8"),
        ("Pred t+10",   f"{horizon_mses.get(10,float('nan')):.5f}", "#ce93d8"),
    ]
    for i, (label, val, color) in enumerate(lines):
        ax5.text(0.05, 0.93 - i * 0.11, label, transform=ax5.transAxes,
                 color=color, fontsize=10,
                 fontweight="bold" if label == "SCORECARD" else "normal")
        ax5.text(0.55, 0.93 - i * 0.11, val, transform=ax5.transAxes,
                 color="white", fontsize=10)

    fig.suptitle("Évaluation AutoEncoder baseline", color="white", fontsize=13, y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK)
        print(f"\nFigure sauvegardée : {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = get_device()
    print(f"Device : {device}")
    model = load_model(args.checkpoint, device)
    train_loader, val_loader = make_loaders(args.dataset_dir)
    print(f"Train : {len(train_loader.dataset)} traj  |  Val : {len(val_loader.dataset)} traj")

    r2s, r2_global, preds, s_val, r2s_mlp, r2_mlp = linear_probe(
        model, train_loader, val_loader, device, n_epochs=args.probe_epochs)
    _, _, seqs_train = collect_embeddings(model, train_loader, device, normalize=True)
    _, _, seqs_val   = collect_embeddings(model, val_loader,   device, normalize=True)
    uniformity, alignment = uniformity_alignment(seqs_train, seqs_val)
    horizon_mses = prediction_horizon(model, val_loader, device, horizons=args.horizons)
    recon_mse    = reconstruction_quality(model, val_loader, device)

    print("\n── Résumé ────────────────────────────────────────────────")
    print(f"  R² global (linéaire) : {r2_global:.4f}")
    print(f"  R² global (MLP)      : {r2_mlp:.4f}")
    print(f"  Uniformité           : {uniformity:.4f}")
    print(f"  Reconstruction MSE   : {recon_mse:.6f}")

    save_figure(r2s, r2_global, uniformity, alignment, horizon_mses,
                recon_mse, preds, s_val, args.save, r2s_mlp=r2s_mlp, r2_mlp=r2_mlp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--dataset-dir",  default="dataset/pendulum")
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--horizons",     type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--save",         default=None)
    main(parser.parse_args())
