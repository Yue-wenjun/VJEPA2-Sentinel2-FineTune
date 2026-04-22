"""
6-channel PatchEmbed3D for Sentinel-2, initialized from V-JEPA 2.1 pretrained weights.

Strategy (Prithvi-style):
  1. Load pretrained Conv3d weight: [hidden_dim, 3, tubelet_size, patch_size, patch_size]
  2. Average across the 3 RGB channels → [hidden_dim, 1, t, p, p]
  3. Repeat 6 times → [hidden_dim, 6, t, p, p]
  4. Copy bias unchanged

This is far better than random init because:
  - Low-frequency spatial features (edges, textures) transfer well
  - The backbone weights remain valid immediately at fine-tuning step 0
"""

import torch
import torch.nn as nn
from einops import rearrange


class PatchEmbed3D_6ch(nn.Module):
    """
    6-channel video patch embedding for S2 (B02 B03 B04 B08 B11 B12).
    Structurally identical to V-JEPA's PatchEmbed3D but in_chans=6.
    """

    def __init__(self, patch_size: int = 16, tubelet_size: int = 2, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_channels=6,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        # x: [B, 6, T, H, W]
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def build_6ch_patch_embed_from_pretrained(
    pretrained_state_dict: dict,
    patch_size: int = 16,
    tubelet_size: int = 2,
    embed_dim: int = 768,
    weight_key: str = "patch_embed.proj.weight",
    bias_key:   str = "patch_embed.proj.bias",
) -> PatchEmbed3D_6ch:
    """
    Build a PatchEmbed3D_6ch and initialize its weights from a pretrained
    3-channel V-JEPA checkpoint.

    Args:
        pretrained_state_dict: loaded checkpoint dict (torch.load(...))
        weight_key: key for Conv3d weight in state dict
        bias_key:   key for Conv3d bias in state dict

    Returns:
        Initialized PatchEmbed3D_6ch module
    """
    module = PatchEmbed3D_6ch(patch_size=patch_size, tubelet_size=tubelet_size, embed_dim=embed_dim)

    if weight_key in pretrained_state_dict:
        w3 = pretrained_state_dict[weight_key]     # [D, 3, t, p, p]
        w_mean = w3.mean(dim=1, keepdim=True)      # [D, 1, t, p, p] -- average over RGB
        w6 = w_mean.repeat(1, 6, 1, 1, 1)         # [D, 6, t, p, p]
        module.proj.weight = nn.Parameter(w6)
        print(f"  patch_embed: initialized 6ch from averaged RGB weights {w3.shape} → {w6.shape}")
    else:
        print(f"  WARNING: key '{weight_key}' not found in checkpoint; using random init")

    if bias_key in pretrained_state_dict:
        module.proj.bias = nn.Parameter(pretrained_state_dict[bias_key].clone())

    return module


# --- DOY positional encoding --------------------------------------------------

class DOYEncoding(nn.Module):
    """
    Injects day-of-year as an additive learned bias on the temporal axis.

    The sinusoidal DOY vector (dim=doy_dim) is projected to embed_dim
    and added to every spatial token of the corresponding frame.

    Usage:
        doy_enc = DOYEncoding(embed_dim=768)
        # During forward pass, after patch embedding:
        # x: [B, N_tokens, D]   where N_tokens = T/tubelet * H/patch * W/patch
        # doys: [B, T] integer day-of-year per frame
        x = x + doy_enc(doys, tubelet_size=2, n_spatial_tokens=196)
    """

    DOY_DIM = 4  # sin/cos at 2 frequencies

    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Linear(self.DOY_DIM, embed_dim, bias=False)

    @staticmethod
    def encode_doy(doys: torch.Tensor) -> torch.Tensor:
        """doys: [B, T] int → [B, T, DOY_DIM] float"""
        import math
        d = doys.float()
        enc = torch.stack([
            torch.sin(2 * math.pi * d / 365),
            torch.cos(2 * math.pi * d / 365),
            torch.sin(4 * math.pi * d / 365),
            torch.cos(4 * math.pi * d / 365),
        ], dim=-1)
        return enc

    def forward(
        self,
        doys: torch.Tensor,        # [B, T]
        tubelet_size: int,
        n_spatial_tokens: int,     # H/patch * W/patch
    ) -> torch.Tensor:
        """Returns bias tensor [B, N_tokens, D] to add to patch embeddings."""
        B, T = doys.shape
        T_tube = T // tubelet_size

        enc = self.encode_doy(doys)                    # [B, T, 4]
        # Aggregate DOY per tubelet (mean of tubelet_size frames)
        enc = enc.view(B, T_tube, tubelet_size, self.DOY_DIM).mean(dim=2)  # [B, T_tube, 4]
        enc = self.proj(enc)                           # [B, T_tube, D]
        # Expand to all spatial tokens
        enc = enc.unsqueeze(2).expand(-1, -1, n_spatial_tokens, -1)  # [B, T_tube, S, D]
        enc = rearrange(enc, "b t s d -> b (t s) d")  # [B, N_tokens, D]
        return enc
