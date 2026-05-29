import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt

from .encoder import ContextEncoder


class TransitionPredictor(nn.Module):
    """MLP de transition : z_t → ẑ_{t+1}  (même architecture que lewm.py)."""

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
        # L2-normalize la sortie : empêche la dérive de norme lors des longs rollouts
        return F.normalize(z + self.net(z), dim=-1)


class AEDecoder(nn.Module):
    """
    Décodeur z → frame  (symétrique de ContextEncoder).

    Pas de normalisation L2 en entrée : le gradient doit traverser la magnitude
    de z pour que l'encodeur apprenne à la contrôler.
    Architecture miroir : FC → reshape (256,4,4) → ConvTranspose ×4 → (3,64,64).
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 256 * 4 * 4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 4→8
            nn.ReLU(),
            nn.ConvTranspose2d(128,  64, 4, stride=2, padding=1),  # 8→16
            nn.ReLU(),
            nn.ConvTranspose2d( 64,  32, 4, stride=2, padding=1),  # 16→32
            nn.ReLU(),
            nn.ConvTranspose2d( 32,   3, 4, stride=2, padding=1),  # 32→64
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D) ou (B, T, D) → frame: (B, 3, 64, 64) ou (B, T, 3, 64, 64)"""
        seq = z.dim() == 3
        if seq:
            B, T, D = z.shape
            z = z.reshape(B * T, D)

        x = self.fc(z).view(z.shape[0], 256, 4, 4)
        out = self.deconv(x)

        if seq:
            out = out.view(B, T, 3, 64, 64)
        return out


class AutoEncoder(nn.Module):
    """
    Baseline autoencoder pour comparaison avec LeWorldModel (JEPA).

    Différence fondamentale vs JEPA :
      - JEPA : supervision dans l'espace latent (target encoder EMA)
      - AE   : supervision dans l'espace pixel (reconstruction MSE)

    Pipeline (entraînement) :
      1. frame stacking        : (frame_t, diff_t) → 6 canaux
      2. Encoder               : 6ch → z_t  (B, T, D)   [gradient actif]
      3. Predictor (k fois)    : z_t → ẑ_{t+k}          [rollout k=1…rollout_k]
      4. Decoder               : ẑ_{t+k} → framê_{t+k}  [reconstruction pixel]
      5. Loss                  : MSE(framê_{t+k}, frame_{t+k})

    Pas d'encodeur cible EMA, pas de SIGReg, pas de cosine loss.
    Encoder + predictor + decoder sont entraînés conjointement.
    """

    def __init__(
        self,
        embed_dim:      int   = 128,
        hidden_dim:     int   = 512,
        rollout_k:      int   = 20,
        rollout_gamma:  float = 0.9,
    ):
        super().__init__()
        self.embed_dim     = embed_dim
        self.rollout_k     = rollout_k
        self.rollout_gamma = rollout_gamma

        self.encoder   = ContextEncoder(embed_dim, in_channels=6)
        self.predictor = TransitionPredictor(embed_dim, hidden_dim)
        self.decoder   = AEDecoder(embed_dim)

    # ── Frame stacking ───────────────────────────────────────────────────────

    @staticmethod
    def _make_pairs(frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) → (B, T, 6, H, W) : concat(frame_t, frame_t - frame_{t-1})"""
        diff = torch.zeros_like(frames)
        diff[:, 1:] = frames[:, 1:] - frames[:, :-1]
        return torch.cat([frames, diff], dim=2)

    # ── Forward (entraînement) ───────────────────────────────────────────────

    @staticmethod
    def _wmse(pred: torch.Tensor, target: torch.Tensor, pw: float) -> torch.Tensor:
        """MSE pondérée : pixels brillants (pendule) reçoivent un poids (1 + pw * target).
        Corrige le déséquilibre fond-noir / tige-blanche (~1-2 % des pixels)."""
        w = 1.0 + pw * target
        return (w * (pred - target).pow(2)).mean()

    def forward(self, frames: torch.Tensor,
                var_lambda: float = 0.1,
                pixel_weight: float = 10.0) -> dict:
        """
        Args:
            frames       : (B, T, 3, H, W)  normalisées [0, 1]
            var_lambda   : poids régularisation variance (anti-collapse)
            pixel_weight : sur-pondération pixels brillants dans la MSE

        Returns:
            dict : loss, recon_loss, var_loss (scalaires)
        """
        B, T, C, H, W = frames.shape
        pairs = self._make_pairs(frames)
        # Normalisation L2 : ancre l'espace latent sur la sphère unité
        # → la norme de z ne peut pas dériver lors des rollouts
        z = F.normalize(
            self.encoder(pairs.reshape(B * T, 6, H, W)).view(B, T, self.embed_dim),
            dim=-1,
        )

        # k=0 : reconstruction directe (poids 1.0)
        weight_sum = 1.0
        recon_loss = self._wmse(self.decoder(z), frames, pixel_weight)

        # k=1..rollout_k : prédiction future avec discount exponentiel γ^k
        # Gradient complet sur l'encodeur pour tous les k : l'encodeur doit
        # apprendre à produire des z stables sous rollout (plus de grad_cutoff).
        # La normalisation L2 du predictor borne les gradients naturellement.
        gamma = self.rollout_gamma
        w = gamma
        for k in range(1, self.rollout_k + 1):
            T_k    = T - k
            z_roll = z[:, :T_k]
            for _ in range(k):
                # Gradient checkpointing : stocke seulement l'entrée de chaque
                # appel, recompute les activations internes pendant le backward.
                # Réduit la mémoire de O(rollout_k²) à O(rollout_k) sans
                # couper le gradient (contrairement à grad_cutoff).
                if self.training:
                    z_roll = grad_ckpt(self.predictor, z_roll, use_reentrant=False)
                else:
                    z_roll = self.predictor(z_roll)
            if self.training:
                frame_pred = grad_ckpt(self.decoder, z_roll, use_reentrant=False)
            else:
                frame_pred = self.decoder(z_roll)
            frame_tgt  = frames[:, k:k + T_k]     # (B, T_k, 3, H, W)
            recon_loss = recon_loss + w * self._wmse(frame_pred, frame_tgt, pixel_weight)
            weight_sum += w
            w *= gamma
        recon_loss = recon_loss / weight_sum

        # Régularisation variance (VICReg) : std de chaque dim >= 1 sur le batch
        # Évite le collapse en pénalisant les embeddings constants
        z_flat  = z.reshape(-1, self.embed_dim)          # (B*T, D)
        var_loss = F.relu(1.0 - z_flat.var(dim=0).sqrt()).mean()

        loss = recon_loss + var_lambda * var_loss

        return {
            "loss":       loss,
            "recon_loss": recon_loss.detach(),
            "var_loss":   var_loss.detach(),
        }

    # ── Inférence ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 3, H, W) → z: (B, T, embed_dim)  L2-normé"""
        B, T, C, H, W = frames.shape
        pairs = self._make_pairs(frames)
        z = self.encoder(pairs.reshape(B * T, 6, H, W)).view(B, T, self.embed_dim)
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def imagine(self, z0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Rollout latent depuis z0.

        Args:
            z0      : (B, embed_dim) ou (B, 1, embed_dim)
            n_steps : nombre de pas

        Returns:
            z_traj : (B, n_steps + 1, embed_dim)
        """
        if z0.dim() == 2:
            z0 = z0.unsqueeze(1)

        traj = [z0]
        for _ in range(n_steps):
            traj.append(self.predictor(traj[-1]))

        return torch.cat(traj, dim=1)
