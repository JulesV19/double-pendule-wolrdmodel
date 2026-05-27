import torch
import torch.nn as nn


class Decoder(nn.Module):
    """
    Symétrique de ContextEncoder : z ∈ R^embed_dim → frame ∈ [0,1]^(3, 64, 64).

    Architecture miroir du CNN encoder :
      FC → reshape (256, 4, 4)
      → ConvTranspose ×4 → (3, 64, 64)
    """

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

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
        """
        z : (B, embed_dim)  ou  (B, T, embed_dim)
        Returns frame : (B, 3, 64, 64)  ou  (B, T, 3, 64, 64)
        """
        seq = z.dim() == 3
        if seq:
            B, T, D = z.shape
            z = z.reshape(B * T, D)

        x = self.fc(z).view(z.shape[0], 256, 4, 4)
        out = self.deconv(x)                           # (B(*T), 3, 64, 64)

        if seq:
            out = out.view(B, T, 3, 64, 64)
        return out
