"""
Recurrent State Space Model (RSSM) — DreamerV2 style, sans actions.

Architecture :
    h_t = GRUCell(s_{t-1}, h_{t-1})        état déterministe (mémoire)
    s_t ~ q(s_t | h_t, enc(o_t))            posterior   (training)
    s_t ~ p(s_t | h_t)                      prior       (imagination)
    o_t ~ decode(cat(h_t, s_t))             reconstruction pixel

État latent complet : z_t = cat(h_t, s_t)   dim = h_dim + s_dim
Loss = wmse pixel + kl_scale * KL(posterior ∥ prior)  avec free-nats

Différence fondamentale vs LeWorldModel (JEPA) :
  RSSM : supervision pixel + KL divergence — décodeur dans la boucle d'entraînement
  JEPA : supervision cosine dans l'espace latent — pas de décodeur pendant l'entraînement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import ContextEncoder
from .ae import AEDecoder


# ── KL analytique ────────────────────────────────────────────────────────────────

def _kl_gaussian(mu_q, std_q, mu_p, std_p):
    """KL(N(mu_q, std_q²) ∥ N(mu_p, std_p²)), somme sur last dim. Retourne (B,)."""
    return (
        torch.log(std_p / std_q)
        + (std_q ** 2 + (mu_q - mu_p) ** 2) / (2.0 * std_p ** 2)
        - 0.5
    ).sum(dim=-1)


def _wmse(pred, target, pw):
    """MSE pondérée : pixels brillants (pendule) reçoivent un poids (1 + pw * target)."""
    w = 1.0 + pw * target
    return (w * (pred - target).pow(2)).mean()


# ── Composants ───────────────────────────────────────────────────────────────────

class _Prior(nn.Module):
    """p(s_t | h_t) → (μ, σ)   —   utilisé pendant l'imagination."""

    def __init__(self, h_dim: int, s_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim, hidden), nn.ELU(),
            nn.Linear(hidden, 2 * s_dim),
        )

    def forward(self, h: torch.Tensor):
        mu, std_param = self.net(h).chunk(2, dim=-1)
        return mu, F.softplus(std_param) + 0.1   # σ ≥ 0.1 pour stabilité


class _Posterior(nn.Module):
    """q(s_t | h_t, feat_t) → (μ, σ)   —   utilisé pendant l'entraînement."""

    def __init__(self, h_dim: int, feat_dim: int, s_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + feat_dim, hidden), nn.ELU(),
            nn.Linear(hidden, 2 * s_dim),
        )

    def forward(self, h: torch.Tensor, feat: torch.Tensor):
        mu, std_param = self.net(torch.cat([h, feat], dim=-1)).chunk(2, dim=-1)
        return mu, F.softplus(std_param) + 0.1


# ── Modèle principal ─────────────────────────────────────────────────────────────

class RSSM(nn.Module):
    """
    RSSM baseline pour comparer avec LeWorldModel (JEPA).

    Paramètres :
        feat_dim   : sortie de l'encodeur CNN (= embed_dim de l'AE)
        h_dim      : taille de l'état déterministe (GRU hidden)
        s_dim      : taille de l'état stochastique
        hidden_dim : taille des MLP prior / posterior
    """

    def __init__(
        self,
        feat_dim:   int = 128,
        h_dim:      int = 200,
        s_dim:      int = 32,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.feat_dim   = feat_dim
        self.h_dim      = h_dim
        self.s_dim      = s_dim
        self.latent_dim = h_dim + s_dim   # dimension du vecteur envoyé au décodeur

        self.encoder   = ContextEncoder(feat_dim, in_channels=6)
        # Projection du state stochastique avant le GRU (PlaNet/DreamerV2 :
        # fc_state_action). Sans action : Linear(s_dim → h_dim) + ELU.
        # Permet au GRU de recevoir une représentation de bonne dimension
        # et améliore le gradient flow depuis s vers h.
        self.fc_embed  = nn.Sequential(nn.Linear(s_dim, h_dim), nn.ELU())
        self.gru_cell  = nn.GRUCell(h_dim, h_dim)   # input_size=h_dim après projection
        self.prior     = _Prior(h_dim, s_dim, hidden_dim)
        self.posterior = _Posterior(h_dim, feat_dim, s_dim, hidden_dim)
        # Réutilise AEDecoder avec embed_dim = h_dim + s_dim
        self.decoder   = AEDecoder(h_dim + s_dim)

    # ── Utilitaires ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_pairs(frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) → (B, T, 6, H, W) : concat(frame_t, frame_t - frame_{t-1})"""
        diff = torch.zeros_like(frames)
        diff[:, 1:] = frames[:, 1:] - frames[:, :-1]
        return torch.cat([frames, diff], dim=2)

    def _init_state(self, B: int, device):
        h = torch.zeros(B, self.h_dim, device=device)
        s = torch.zeros(B, self.s_dim, device=device)
        return h, s

    # ── Forward (entraînement) ───────────────────────────────────────────────────

    def forward(
        self,
        frames:       torch.Tensor,
        kl_scale:     float = 1.0,
        pixel_weight: float = 10.0,
        free_nats:    float = 1.0,
    ) -> dict:
        """
        Args:
            frames       : (B, T, 3, H, W)  normalisées [0, 1]
            kl_scale     : poids du terme KL dans la loss totale
            pixel_weight : sur-pondération pixels brillants dans wmse
            free_nats    : plancher KL par pas de temps (évite sur-pénalisation initiale)

        Returns:
            dict : loss, recon_loss, kl_loss  (scalaires)
        """
        B, T, C, H, W = frames.shape
        device = frames.device

        # Encodage de toutes les frames en un seul appel CNN (parallèle sur B*T)
        pairs = self._make_pairs(frames)   # (B, T, 6, H, W)
        feats = self.encoder(
            pairs.reshape(B * T, 6, H, W)
        ).view(B, T, self.feat_dim)        # (B, T, feat_dim)

        h, s = self._init_state(B, device)
        h_list, s_list, kl_list = [], [], []

        for t in range(T):
            # Embedding du state stochastique → même dim que h avant le GRU
            x = self.fc_embed(s)           # (B, h_dim)
            h = self.gru_cell(x, h)        # h_t = GRU(fc_embed(s_{t-1}), h_{t-1})

            mu_p, std_p = self.prior(h)
            mu_q, std_q = self.posterior(h, feats[:, t])

            # Reparameterization trick — s vient du posterior pendant l'entraînement
            s = mu_q + std_q * torch.randn_like(mu_q)

            h_list.append(h)
            s_list.append(s)
            kl_list.append(_kl_gaussian(mu_q, std_q, mu_p, std_p))   # (B,)

        h_seq = torch.stack(h_list, dim=1)            # (B, T, h_dim)
        s_seq = torch.stack(s_list, dim=1)            # (B, T, s_dim)
        z_seq = torch.cat([h_seq, s_seq], dim=-1)     # (B, T, latent_dim)

        # Reconstruction pixel
        recon_loss = _wmse(self.decoder(z_seq), frames, pixel_weight)

        # KL brute (non-clamped) — utile pour diagnostiquer le posterior collapse
        kl_raw = torch.stack(kl_list, dim=1).mean()   # (B, T) → scalaire

        # Free-nats sur le scalaire final : gradient nul quand KL < free_nats
        # (PlaNet/DreamerV2 : max(mean_KL, free_nats))
        kl_loss = torch.clamp(kl_raw, min=free_nats)

        loss = recon_loss + kl_scale * kl_loss
        return {
            "loss":       loss,
            "recon_loss": recon_loss.detach(),
            "kl_loss":    kl_loss.detach(),
            "kl_raw":     kl_raw.detach(),   # KL réelle avant clamp — diagnostic collapse
        }

    # ── Inférence ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode une séquence via le posterior (moyenne — déterministe).

        frames : (B, T, 3, H, W)
        Retourne z : (B, T, latent_dim)  utilisable pour les probes linéaires.
        """
        B, T, C, H, W = frames.shape
        pairs = self._make_pairs(frames)
        feats = self.encoder(
            pairs.reshape(B * T, 6, H, W)
        ).view(B, T, self.feat_dim)

        h, s = self._init_state(B, frames.device)
        z_list = []

        for t in range(T):
            h = self.gru_cell(self.fc_embed(s), h)
            mu_q, _ = self.posterior(h, feats[:, t])
            s = mu_q   # moyenne (pas d'échantillonnage pour l'encodage)
            z_list.append(torch.cat([h, s], dim=-1))

        return torch.stack(z_list, dim=1)   # (B, T, latent_dim)

    @torch.no_grad()
    def imagine(
        self,
        frames_seed: torch.Tensor,
        n_steps:     int,
        stochastic:  bool = False,
    ) -> torch.Tensor:
        """
        Rollout via le prior depuis des frames réelles d'amorçage.
        L'encodeur n'est plus utilisé après les frames de graine.

        Args:
            frames_seed : (B, T_seed, 3, H, W)  frames réelles d'amorçage
            n_steps     : nombre de steps à imaginer au-delà de T_seed
            stochastic  : si True, sample du prior ; sinon utilise la moyenne (μ_p)

        Returns:
            z_traj : (B, T_seed + n_steps, latent_dim)  — décodable via self.decoder
        """
        B, T_seed, C, H, W = frames_seed.shape
        device = frames_seed.device

        # Phase de graine — posterior sur les frames réelles
        pairs = self._make_pairs(frames_seed)
        feats = self.encoder(
            pairs.reshape(B * T_seed, 6, H, W)
        ).view(B, T_seed, self.feat_dim)

        h, s = self._init_state(B, device)
        z_list = []

        for t in range(T_seed):
            h = self.gru_cell(self.fc_embed(s), h)
            mu_q, _ = self.posterior(h, feats[:, t])
            s = mu_q
            z_list.append(torch.cat([h, s], dim=-1))

        # Phase d'imagination — prior uniquement, sans encodeur
        for _ in range(n_steps):
            h = self.gru_cell(self.fc_embed(s), h)
            mu_p, std_p = self.prior(h)
            s = mu_p + std_p * torch.randn_like(mu_p) if stochastic else mu_p
            z_list.append(torch.cat([h, s], dim=-1))

        return torch.stack(z_list, dim=1)   # (B, T_seed + n_steps, latent_dim)
