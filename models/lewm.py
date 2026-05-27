import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ContextEncoder, TargetEncoder
from .sigreg  import sigreg_loss


class TransitionPredictor(nn.Module):
    """
    MLP de transition : z_t → ẑ_{t+1}.

    Voit UNIQUEMENT z_t — pas de contexte séquentiel.
    Cela force l'encodeur à mettre θ ET ω dans z_t : sans ω, impossible
    de prédire z_{t+1} depuis z_t seul (le bras peut partir dans n'importe
    quelle direction depuis la même position).

    Architecture résiduelle pour faciliter l'apprentissage de petits δz.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, D) → ẑ: (B, T, D)  —  ẑ[:, t] prédit z[:, t+1]"""
        return self.net(z)


class LeWorldModel(nn.Module):
    """
    LeWorldModel — JEPA pur, sans supervision d'état.

    Pipeline :
      1. Encoder online  : (frame_t, diff_t) → z_ctx   (B, T, D)  [gradient actif]
      2. Encoder target  : (frame_t, diff_t) → z_tgt   (B, T, D)  [EMA, no gradient]
      3. Predictor MLP   : z_ctx_t → ẑ_{t+1}           (B, T-1, D)
      4. Pred loss       : MSE(ẑ_{t+1}, z_tgt_{t+1})
      5. SIGReg          : force z_ctx ~ N(0, I)

    Pourquoi le MLP force ω dans z (vs Transformer causal) :
      Le Transformer voyait z_{0..t} et pouvait calculer z_t - z_{t-1} ≈ ω
      lui-même, sans que l'encodeur ait besoin de l'encoder. Le MLP ne voit
      que z_t : sans ω dans z_t, la prédiction de z_{t+1} est impossible.

    Appeler model.update_target() après chaque optimizer.step().
    """

    def __init__(
        self,
        embed_dim:    int   = 128,
        hidden_dim:   int   = 512,
        lam:          float = 0.1,
        n_proj:       int   = 512,
        ema_momentum: float = 0.996,
        rollout_k:    int   = 2,     # steps de prédiction pour forcer ω dans z
        # conservés pour compatibilité CLI mais non utilisés
        n_heads:      int   = 4,
        n_layers:     int   = 4,
        max_frames:   int   = 64,
        mask_ratio:   float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.lam       = lam
        self.n_proj    = n_proj
        self.rollout_k = rollout_k

        self.encoder        = ContextEncoder(embed_dim, in_channels=6)
        self.target_encoder = TargetEncoder(self.encoder, momentum=ema_momentum)
        self.predictor      = TransitionPredictor(embed_dim, hidden_dim)

    # ── Frame stacking ───────────────────────────────────────────────────────

    @staticmethod
    def _make_pairs(frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) → (B, T, 6, H, W) : concat(frame_t, frame_t - frame_{t-1})"""
        diff = torch.zeros_like(frames)
        diff[:, 1:] = frames[:, 1:] - frames[:, :-1]
        return torch.cat([frames, diff], dim=2)

    # ── Forward (entraînement) ───────────────────────────────────────────────

    def forward(self, frames: torch.Tensor) -> dict:
        """
        Args:
            frames : (B, T, 3, H, W)  séquences normalisées [0, 1]

        Returns:
            dict : loss, pred_loss, sigreg (scalaires)
        """
        B, T, C, H, W = frames.shape
        pairs      = self._make_pairs(frames)
        frames_flat = pairs.reshape(B * T, 6, H, W)

        z_ctx = self.encoder(frames_flat).view(B, T, self.embed_dim)        # (B, T, D)
        z_tgt = self.target_encoder(frames_flat).view(B, T, self.embed_dim) # (B, T, D)

        # Rollout K steps : z_t → z_{t+K} en enchaînant le predictor
        # Force ω dans z : sans vitesse, les erreurs s'accumulent sur K steps.
        # On utilise la cosine distance (pas MSE) pour éviter le raccourci "moyenne
        # des deux modes ±ω" : outputer la moyenne de deux directions opposées donne
        # cos-sim ≈ 0 (worst case), ce qui force le predictor à choisir une direction.
        K      = self.rollout_k
        z_roll = z_ctx[:, :T - K]          # (B, T-K, D) — points de départ
        for _ in range(K):
            z_roll = self.predictor(z_roll)
        z_roll_n  = F.normalize(z_roll,           dim=-1)
        z_tgt_n   = F.normalize(z_tgt[:, K:],     dim=-1)
        pred_loss = (1.0 - (z_roll_n * z_tgt_n).sum(dim=-1)).mean()

        z_flat = z_ctx.reshape(B * T, self.embed_dim)
        sigreg = sigreg_loss(z_flat, self.n_proj)

        loss = pred_loss + self.lam * sigreg

        return {
            "loss":      loss,
            "pred_loss": pred_loss.detach(),
            "sigreg":    sigreg.detach(),
        }

    def update_target(self) -> None:
        self.target_encoder.update(self.encoder)

    # ── Inférence ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 3, H, W) → z: (B, T, embed_dim)"""
        B, T, C, H, W = frames.shape
        pairs = self._make_pairs(frames)
        z = self.encoder(pairs.reshape(B * T, 6, H, W))
        return z.view(B, T, self.embed_dim)

    @torch.no_grad()
    def imagine(self, z0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Rollout depuis z0 en appliquant le MLP de transition pas à pas.

        Args:
            z0      : (B, embed_dim) ou (B, 1, embed_dim)
            n_steps : nombre de pas à prédire

        Returns:
            z_traj : (B, n_steps + 1, embed_dim)  — inclut z0
        """
        if z0.dim() == 2:
            z0 = z0.unsqueeze(1)                          # (B, 1, D)

        traj = [z0]
        for _ in range(n_steps):
            z_next = self.predictor(traj[-1])             # (B, 1, D) → (B, 1, D)
            traj.append(z_next)

        return torch.cat(traj, dim=1)                     # (B, n_steps+1, D)
