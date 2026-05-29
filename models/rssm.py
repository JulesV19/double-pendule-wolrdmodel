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
        # Le décodeur utilise UNIQUEMENT h_t (état déterministe).
        # Décoder depuis cat(h,s) crée un "posterior shortcut" : s_t peut encoder
        # la frame courante directement, le GRU n'apprend pas de dynamique, et
        # l'imagination échoue (anneau blanc). Décoder depuis h_t force le GRU
        # à porter toute l'information temporelle.
        self.latent_dim = h_dim   # = dim décodeur = dim retournée par encode()/imagine()

        self.encoder   = ContextEncoder(feat_dim, in_channels=6)
        self.fc_embed  = nn.Sequential(nn.Linear(s_dim, h_dim), nn.ELU())
        # fc_trans : transition autonome h_t → input GRU, utilisée pendant
        # l'imagination ET pendant les steps "free-running" du scheduled sampling.
        # Entraîner fc_trans pendant le training réduit le biais d'exposition :
        # le GRU apprend à avancer ses dynamiques depuis son propre état.
        # (Bengio et al. 2015, "Scheduled Sampling for Sequence Prediction")
        self.fc_trans  = nn.Sequential(nn.Linear(h_dim, h_dim), nn.ELU())
        self.gru_cell  = nn.GRUCell(h_dim, h_dim)
        self.prior     = _Prior(h_dim, s_dim, hidden_dim)
        self.posterior = _Posterior(h_dim, feat_dim, s_dim, hidden_dim)
        self.decoder   = AEDecoder(h_dim)   # entrée = h_t uniquement

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
        ss_rate:      float = 0.0,
    ) -> dict:
        """
        Args:
            frames       : (B, T, 3, H, W)  normalisées [0, 1]
            kl_scale     : poids du terme KL
            pixel_weight : sur-pondération pixels brillants dans wmse
            free_nats    : plancher KL sur le scalaire final
            ss_rate      : probabilité de free-running à ce step d'entraînement
                           (scheduled sampling, augmente de 0 → ss_max_rate au fil des epochs)

        Returns:
            dict : loss, recon_loss, kl_loss, kl_raw  (scalaires)
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
            # Scheduled sampling : avec probabilité ss_rate, utiliser fc_trans(h)
            # (transition autonome, identique à l'imagination) au lieu du posterior.
            # Augmenter ss_rate au fil des epochs réduit le biais d'exposition.
            use_free_run = (
                self.training
                and ss_rate > 0.0
                and torch.rand(1, device=frames.device).item() < ss_rate
            )

            if use_free_run:
                x = self.fc_trans(h)                          # self-transition
                # posterior calculé quand même pour la KL (monitoring)
                mu_q, std_q = self.posterior(h, feats[:, t])
            else:
                mu_q, std_q = self.posterior(h, feats[:, t]) # teacher forcing
                x = self.fc_embed(mu_q)

            h = self.gru_cell(x, h)

            mu_p, std_p = self.prior(h)
            s = mu_q + std_q * torch.randn_like(mu_q)

            h_list.append(h)
            s_list.append(s)
            kl_list.append(_kl_gaussian(mu_q, std_q, mu_p, std_p))   # (B,)

        h_seq = torch.stack(h_list, dim=1)   # (B, T, h_dim)

        # Reconstruction depuis h_t uniquement — force le GRU à apprendre les dynamiques.
        # s_t reste dans la loss via KL mais n'a pas accès au décodeur.
        recon_loss = _wmse(self.decoder(h_seq), frames, pixel_weight)

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
            s = mu_q
            z_list.append(h)   # h uniquement — cohérent avec le décodeur

        return torch.stack(z_list, dim=1)   # (B, T, h_dim)

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

        # Phase de graine : teacher forcing (posterior sur les frames réelles)
        s = torch.zeros(B, self.s_dim, device=device)
        for t in range(T_seed):
            mu_q, _ = self.posterior(h, feats[:, t])
            h = self.gru_cell(self.fc_embed(mu_q), h)
            s = mu_q
            z_list.append(h)

        # Phase d'imagination : fc_trans (même chemin que le free-running du training)
        # Plus de prior needed — fc_trans est entraîné explicitement pour cette tâche.
        for _ in range(n_steps):
            h = self.gru_cell(self.fc_trans(h), h)
            z_list.append(h)

        return torch.stack(z_list, dim=1)   # (B, T_seed + n_steps, h_dim)
