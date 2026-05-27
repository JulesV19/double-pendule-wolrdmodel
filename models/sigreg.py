import math
import torch
import torch.nn as nn


def sigreg_loss(z: torch.Tensor, n_proj: int = 512, max_n: int = 256) -> torch.Tensor:
    """
    SIGReg : Statistical Isotropic Gaussian Regularizer (LeWorldModel, 2026).

    Principe (théorème de Cramér-Wold) :
      Une distribution D^d est gaussienne isotrope N(0, I) si et seulement si
      toute projection 1D sur un vecteur aléatoire u ~ Uniforme(S^{d-1})
      suit une loi N(0, 1).

    SIGReg projette les embeddings sur n_proj directions aléatoires et applique
    le test de normalité d'Epps-Pulley à chaque projection 1D.
    Minimiser SIGReg pousse la distribution des embeddings vers N(0, I).

    Complexité : O(N² × n_proj) — raisonnable pour N < 1000, n_proj ≤ 512.

    Référence :
      Maes et al., "LeWorldModel", arXiv:2603.19312, 2026.
      Epps & Pulley, "A test of normality based on the empirical
      characteristic function", Biometrika, 1983.

    Args:
        z      : (N, D) embeddings (non normalisés)
        n_proj : M — nombre de projections aléatoires (défaut 512)

    Returns:
        scalaire ≥ 0  (0 = distribution parfaitement gaussienne isotrope)
    """
    N, D = z.shape

    # Subsample pour borner la mémoire O(N² × n_proj)
    if N > max_n:
        idx = torch.randperm(N, device=z.device)[:max_n]
        z = z[idx]
        N = max_n

    # Directions aléatoires uniformes sur la sphère S^{D-1}
    u = torch.randn(D, n_proj, device=z.device, dtype=z.dtype)
    u = u / u.norm(dim=0, keepdim=True)          # (D, n_proj)

    # Projections 1D : h[n, m] = z[n] · u[m]
    # Pas de standardisation — la formule d'Epps-Pulley compare nativement
    # contre N(0,1), ce qui pénalise les erreurs de moyenne ET de variance.
    # Standardiser ici rendrait SIGReg aveugle au collapse (gradient nul quand
    # tous les embeddings sont identiques, car z - mean → 0 et std → 0).
    h = z @ u                                     # (N, n_proj)

    # ── Statistique d'Epps-Pulley (vectorisée sur toutes les projections) ──
    #
    # T(h) = (1/N²) Σᵢ Σⱼ exp(-(hᵢ-hⱼ)²/2)
    #       - √2 · (1/N) Σᵢ exp(-hᵢ²/4)
    #       + 1/√3
    #
    # Sous H₀ (normalité), T → 0.
    # Si collapse (tous hᵢ identiques) : terme croisé → 1, T → 1/√3 > 0.

    # Terme croisé : (N, N, n_proj) — calculé par blocs si N grand
    diff = h.unsqueeze(0) - h.unsqueeze(1)        # (N, N, n_proj)
    cross = torch.exp(-0.5 * diff.pow(2)).mean(dim=(0, 1))   # (n_proj,)

    # Terme diagonal
    single = torch.exp(-0.25 * h.pow(2)).mean(dim=0)          # (n_proj,)

    T = cross - math.sqrt(2) * single + 1.0 / math.sqrt(3)   # (n_proj,)

    return T.mean()
