import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ContextEncoder
from .sigreg  import sigreg_loss


class CausalPredictor(nn.Module):
    """
    Transformer causal : prédit z_{t+1} à partir de z_0, …, z_t.

    Chaque position ne peut attendre que les positions précédentes (masque
    causal triangulaire inférieur, comme GPT). Il n'y a pas d'encodeur cible
    séparé : encodeur et predictor sont entraînés conjointement.

    Args:
        embed_dim:  dimension des embeddings (= ContextEncoder.embed_dim)
        hidden_dim: dimension feedforward interne
        n_heads:    têtes d'attention
        n_layers:   blocs Transformer
        max_frames: longueur maximale de séquence supportée
    """

    def __init__(
        self,
        embed_dim:  int = 128,
        hidden_dim: int = 512,
        n_heads:    int = 4,
        n_layers:   int = 4,
        max_frames: int = 64,
    ):
        super().__init__()
        self.pos_embed = nn.Embedding(max_frames, embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z : (B, T, embed_dim)

        Returns:
            z_pred : (B, T, embed_dim)  — z_pred[:, t] prédit z[:, t+1]
        """
        T = z.size(1)
        pos = torch.arange(T, device=z.device)
        z = z + self.pos_embed(pos)

        # Masque causal : position i n'attend que 0..i
        mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=z.device, dtype=z.dtype
        )
        out = self.transformer(z, mask=mask, is_causal=True)
        return self.norm(out)


class LeWorldModel(nn.Module):
    """
    LeWorldModel (Maes et al., 2026) — JEPA stable par régularisation SIGReg.

    Différences clés vs JEPA+VICReg :
      • Pas d'encodeur cible EMA — encodeur et predictor entraînés conjointement.
      • Pas de masquage contexte/cible — le predictor est causal (autoregressif).
      • VICReg (6 hyperparamètres) → SIGReg (1 hyperparamètre λ).
      • Anti-collapse garanti mathématiquement par le théorème de Cramér-Wold.

    Forward :
      1. Encoder  : frames → z             (B, T, D)
      2. Predictor: z_{0..T-2} → ẑ_{1..T} (B, T-1, D)  [causal]
      3. Pred loss: MSE(ẑ_t, z_t)          scalaire
      4. SIGReg   : force z ~ N(0, I)      scalaire

    Args:
        embed_dim:  dimension des embeddings
        hidden_dim: dimension feedforward du predictor
        n_heads:    têtes d'attention
        n_layers:   blocs Transformer
        max_frames: longueur maximale de séquence
        lam:        poids SIGReg (λ, seul hyperparamètre effectif)
        n_proj:     projections SIGReg (M, robuste à ce choix)
    """

    def __init__(
        self,
        embed_dim:  int   = 128,
        hidden_dim: int   = 512,
        n_heads:    int   = 4,
        n_layers:   int   = 4,
        max_frames: int   = 64,
        lam:        float = 0.1,
        n_proj:     int   = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.lam       = lam
        self.n_proj    = n_proj

        self.encoder   = ContextEncoder(embed_dim)
        self.predictor = CausalPredictor(embed_dim, hidden_dim,
                                         n_heads, n_layers, max_frames)

    # ── Forward (entraînement) ───────────────────────────────────────────────

    def forward(self, frames: torch.Tensor) -> dict:
        """
        Args:
            frames : (B, T, 3, H, W)  séquences normalisées [0, 1]

        Returns:
            dict : loss, pred_loss, sigreg (tous scalaires)
        """
        B, T, C, H, W = frames.shape

        # Encoder toute la séquence (gradient actif)
        z = self.encoder(frames.reshape(B * T, C, H, W))
        z = z.view(B, T, self.embed_dim)           # (B, T, D)

        # Predictor causal : z_{0..T-2} → ẑ_{1..T}
        z_pred = self.predictor(z[:, :-1])         # (B, T-1, D)
        z_tgt  = z[:, 1:].detach()                 # (B, T-1, D)  — stop-gradient sur la cible

        # Prédiction loss
        pred_loss = F.mse_loss(z_pred, z_tgt)

        # SIGReg sur tous les embeddings encodés
        z_flat  = z.reshape(B * T, self.embed_dim)
        sigreg  = sigreg_loss(z_flat, self.n_proj)

        loss = pred_loss + self.lam * sigreg

        return {
            "loss":      loss,
            "pred_loss": pred_loss.detach(),
            "sigreg":    sigreg.detach(),
        }

    # ── Inférence ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode une séquence de frames.

        Args:
            frames : (B, T, 3, H, W)

        Returns:
            z : (B, T, embed_dim)
        """
        B, T, C, H, W = frames.shape
        z = self.encoder(frames.reshape(B * T, C, H, W))
        return z.view(B, T, self.embed_dim)

    @torch.no_grad()
    def imagine(self, z0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Rollout autoregressif depuis z0.

        Args:
            z0      : (B, embed_dim) ou (B, 1, embed_dim)
            n_steps : nombre de pas à prédire

        Returns:
            z_traj : (B, n_steps + 1, embed_dim)  — inclut z0
        """
        if z0.dim() == 2:
            z0 = z0.unsqueeze(1)

        traj = [z0]
        for _ in range(n_steps):
            ctx    = torch.cat(traj, dim=1)               # (B, step+1, D)
            z_next = self.predictor(ctx)[:, -1:]          # (B, 1, D)
            traj.append(z_next)

        return torch.cat(traj, dim=1)                     # (B, n_steps+1, D)
