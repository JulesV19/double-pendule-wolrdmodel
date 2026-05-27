import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ContextEncoder, TargetEncoder
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
      • Encodeur cible EMA (TargetEncoder) — cible stable, anti-collapse structurel.
      • Pas de masquage contexte/cible — le predictor est causal (autoregressif).
      • VICReg (6 hyperparamètres) → SIGReg (1 hyperparamètre λ).
      • Anti-collapse garanti mathématiquement par le théorème de Cramér-Wold.

    Forward :
      1. Encoder online  : frames → z_ctx              (B, T, D)   [gradient actif]
      2. Encoder target  : frames → z_tgt              (B, T, D)   [EMA, no gradient]
      3. Masquage        : mask_ratio des z_ctx remplacés par mask_token
      4. Predictor causal: z_ctx_masqué_{0..T-2} → ẑ  (B, T-1, D)
      5. Pred loss       : MSE(ẑ, z_tgt) sur positions masquées uniquement
      6. SIGReg          : force z_ctx ~ N(0, I)        scalaire

    Appeler model.update_target() après chaque optimizer.step().

    Args:
        embed_dim:  dimension des embeddings
        hidden_dim: dimension feedforward du predictor
        n_heads:    têtes d'attention
        n_layers:   blocs Transformer
        max_frames: longueur maximale de séquence
        lam:        poids SIGReg (λ, seul hyperparamètre effectif)
        n_proj:     projections SIGReg (M, robuste à ce choix)
        ema_momentum: momentum du target encoder (τ, défaut 0.996)
    """

    def __init__(
        self,
        embed_dim:    int   = 128,
        hidden_dim:   int   = 512,
        n_heads:      int   = 4,
        n_layers:     int   = 4,
        max_frames:   int   = 64,
        lam:          float = 0.1,
        n_proj:       int   = 512,
        ema_momentum: float = 0.996,
        mask_ratio:   float = 0.4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.lam       = lam
        self.n_proj    = n_proj

        self.mask_ratio = mask_ratio

        self.encoder        = ContextEncoder(embed_dim, in_channels=6)
        self.target_encoder = TargetEncoder(self.encoder, momentum=ema_momentum)
        self.predictor      = CausalPredictor(embed_dim, hidden_dim,
                                              n_heads, n_layers, max_frames)
        self.mask_token     = nn.Parameter(torch.zeros(embed_dim))
        self.omega_head     = nn.Linear(embed_dim, 2)   # prédit (ω1, ω2)

    # ── Frame stacking ───────────────────────────────────────────────────────

    @staticmethod
    def _make_pairs(frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) → (B, T, 6, H, W) : concat(frame_t, frame_t - frame_{t-1})"""
        diff = torch.zeros_like(frames)
        diff[:, 1:] = frames[:, 1:] - frames[:, :-1]
        return torch.cat([frames, diff], dim=2)

    # ── Forward (entraînement) ───────────────────────────────────────────────

    def forward(self, frames: torch.Tensor, states: torch.Tensor | None = None) -> dict:
        """
        Args:
            frames : (B, T, 3, H, W)  séquences normalisées [0, 1]
            states : (B, T, 4)  [θ1, θ2, ω1, ω2] — optionnel, active la loss auxiliaire ω

        Returns:
            dict : loss, pred_loss, sigreg, omega_loss (tous scalaires)
        """
        B, T, C, H, W = frames.shape
        pairs = self._make_pairs(frames)               # (B, T, 6, H, W)
        frames_flat = pairs.reshape(B * T, 6, H, W)

        # Encodeur online (gradient actif)
        z_ctx = self.encoder(frames_flat).view(B, T, self.embed_dim)         # (B, T, D)

        # Encodeur target EMA (no gradient) — cible stable
        z_tgt = self.target_encoder(frames_flat).view(B, T, self.embed_dim)  # (B, T, D)

        # Masquage aléatoire du contexte d'entrée du predictor
        # On masque z_{0..T-2} ; la cible reste z_tgt_{1..T-1} (EMA, non masqué)
        z_input = z_ctx[:, :-1].clone()            # (B, T-1, D)
        if self.training and self.mask_ratio > 0:
            mask = torch.rand(B, T - 1, device=frames.device) < self.mask_ratio
            z_input[mask] = self.mask_token        # remplace par le token appris
        else:
            mask = None

        # Predictor causal sur le contexte (potentiellement masqué)
        z_pred = self.predictor(z_input)           # (B, T-1, D)

        # Loss uniquement sur les positions masquées (comme V-JEPA)
        # Si pas de masquage (eval/mask_ratio=0) : loss sur tout
        z_target = z_tgt[:, 1:]
        if mask is not None and mask.any():
            pred_loss = F.mse_loss(z_pred[mask], z_target[mask])
        else:
            pred_loss = F.mse_loss(z_pred, z_target)

        # SIGReg sur les embeddings online
        z_flat = z_ctx.reshape(B * T, self.embed_dim)
        sigreg = sigreg_loss(z_flat, self.n_proj)

        # Loss auxiliaire ω — force l'encodeur à encoder la vitesse dans z
        if states is not None:
            omega_true = states[:, :, 2:].to(z_ctx.device)   # (B, T, 2)
            omega_pred = self.omega_head(z_ctx)               # (B, T, 2)
            omega_loss = F.mse_loss(omega_pred, omega_true)
        else:
            omega_loss = torch.zeros(1, device=z_ctx.device).squeeze()

        loss = pred_loss + self.lam * sigreg + 0.1 * omega_loss

        return {
            "loss":       loss,
            "pred_loss":  pred_loss.detach(),
            "sigreg":     sigreg.detach(),
            "omega_loss": omega_loss.detach(),
        }

    def update_target(self) -> None:
        """Mise à jour EMA du target encoder — appeler après chaque optimizer.step()."""
        self.target_encoder.update(self.encoder)

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
        pairs = self._make_pairs(frames)
        z = self.encoder(pairs.reshape(B * T, 6, H, W))
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
