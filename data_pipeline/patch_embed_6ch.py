"""
N-channel PatchEmbed3D for Sentinel-2, initialized from V-JEPA 2.1 pretrained weights.

Strategy (Prithvi-style):
  1. Load pretrained Conv3d weight: [hidden_dim, 3, tubelet_size, patch_size, patch_size]
  2. Average across the 3 RGB channels → [hidden_dim, 1, t, p, p]
  3. Repeat N times → [hidden_dim, N, t, p, p]
  4. Copy bias unchanged

This is far better than random init because:
  - Low-frequency spatial features (edges, textures) transfer well
  - The backbone weights remain valid immediately at fine-tuning step 0
"""

import torch.nn as nn


class PatchEmbed3D_Nch(nn.Module):
    """
    N-channel video patch embedding for multispectral data.
    Structurally identical to V-JEPA's PatchEmbed3D but in_chans=N.
    """

    def __init__(self, in_chans: int = 6, patch_size: int = 16, tubelet_size: int = 2, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x, **kwargs):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# backward-compat alias
PatchEmbed3D_6ch = PatchEmbed3D_Nch


def build_nch_patch_embed_from_pretrained(
    pretrained_state_dict: dict,
    in_chans: int = 6,
    patch_size: int = 16,
    tubelet_size: int = 2,
    embed_dim: int = 768,
    weight_key: str = "patch_embed.proj.weight",
    bias_key:   str = "patch_embed.proj.bias",
) -> PatchEmbed3D_Nch:
    """
    Build a PatchEmbed3D_Nch and initialize its weights from a pretrained
    3-channel V-JEPA checkpoint via Prithvi-style channel averaging.

    Args:
        pretrained_state_dict: loaded checkpoint dict (torch.load(...))
        in_chans: number of input channels for the new patch embed
        weight_key: key for Conv3d weight in state dict
        bias_key:   key for Conv3d bias in state dict

    Returns:
        Initialized PatchEmbed3D_Nch module
    """
    module = PatchEmbed3D_Nch(in_chans=in_chans, patch_size=patch_size, tubelet_size=tubelet_size, embed_dim=embed_dim)

    if weight_key in pretrained_state_dict:
        w3 = pretrained_state_dict[weight_key]       # [D, 3, t, p, p]
        w_mean = w3.mean(dim=1, keepdim=True)         # [D, 1, t, p, p]
        wN = w_mean.repeat(1, in_chans, 1, 1, 1)     # [D, N, t, p, p]
        module.proj.weight = nn.Parameter(wN)
        print(f"  patch_embed: initialized {in_chans}ch from averaged RGB weights {w3.shape} → {wN.shape}")
    else:
        print(f"  WARNING: key '{weight_key}' not found in checkpoint; using random init")

    if bias_key in pretrained_state_dict:
        module.proj.bias = nn.Parameter(pretrained_state_dict[bias_key].clone())

    return module


def build_6ch_patch_embed_from_pretrained(
    pretrained_state_dict: dict,
    patch_size: int = 16,
    tubelet_size: int = 2,
    embed_dim: int = 768,
    weight_key: str = "patch_embed.proj.weight",
    bias_key:   str = "patch_embed.proj.bias",
) -> PatchEmbed3D_Nch:
    """Backward-compat wrapper for build_nch_patch_embed_from_pretrained with in_chans=6."""
    return build_nch_patch_embed_from_pretrained(
        pretrained_state_dict=pretrained_state_dict,
        in_chans=6,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim,
        weight_key=weight_key,
        bias_key=bias_key,
    )
