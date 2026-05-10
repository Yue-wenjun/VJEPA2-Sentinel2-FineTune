# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
from einops import rearrange


class DOYEncoding(nn.Module):
    """
    Day-of-Year positional encoding for satellite time-series.

    Encodes each frame's calendar date as a sinusoidal vector and projects
    it to embed_dim, then adds it to every spatial token of the corresponding
    tubelet.  This gives the model an explicit seasonal signal that the
    standard frame-index temporal PE cannot provide.

    Supported modes:
        sinusoidal: fixed calendar features projected to embed_dim
        learnable: sinusoidal features plus a learnable DOY table
        month_token: learnable month table derived from DOY

    Args:
        embed_dim: token embedding dimension (must match ViT embed_dim)

    Forward:
        doys:            [B, T_full]  int64, one DOY per original frame
        tubelet_size:    int, number of frames per tubelet (e.g. 2)
        n_spatial_tokens: int, H_patches * W_patches

    Returns:
        bias tensor [B, T_tube * n_spatial_tokens, embed_dim] to add to x
    """

    DOY_DIM = 4

    def __init__(self, embed_dim: int, mode: str = "sinusoidal", num_months: int = 12):
        super().__init__()
        if mode not in {"sinusoidal", "learnable", "month_token"}:
            raise ValueError(
                f"Unknown DOY mode={mode!r}; expected sinusoidal, learnable, or month_token"
            )
        self.mode = mode
        self.num_months = num_months
        self.proj = nn.Linear(self.DOY_DIM, embed_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)
        self.day_embed = nn.Embedding(367, embed_dim) if mode == "learnable" else None
        self.month_embed = nn.Embedding(num_months, embed_dim) if mode == "month_token" else None
        if self.day_embed is not None:
            nn.init.normal_(self.day_embed.weight, std=0.02)
        if self.month_embed is not None:
            nn.init.normal_(self.month_embed.weight, std=0.02)

    @staticmethod
    def sincos(doys: torch.Tensor) -> torch.Tensor:
        """doys: [...] int/float → [..., 4] float"""
        d = doys.float()
        return torch.stack([
            torch.sin(2 * math.pi * d / 365),
            torch.cos(2 * math.pi * d / 365),
            torch.sin(4 * math.pi * d / 365),
            torch.cos(4 * math.pi * d / 365),
        ], dim=-1)

    def forward(
        self,
        doys: torch.Tensor,
        tubelet_size: int,
        n_spatial_tokens: int,
    ) -> torch.Tensor:
        B, T_full = doys.shape
        T_tube = T_full // tubelet_size

        if self.mode == "month_token":
            month = ((doys.float() - 1) * self.num_months / 365).long()
            month = month.clamp_(0, self.num_months - 1)
            enc = self.month_embed(month)                           # [B, T_full, D]
            enc = enc.view(B, T_tube, tubelet_size, -1).mean(dim=2) # [B, T_tube, D]
        else:
            enc = self.sincos(doys)                                 # [B, T_full, 4]
            enc = enc.view(B, T_tube, tubelet_size, self.DOY_DIM)
            enc = enc.mean(dim=2)                                   # [B, T_tube, 4]
            enc = self.proj(enc)                                    # [B, T_tube, D]
            if self.day_embed is not None:
                day = doys.long().clamp_(0, 366)
                day_enc = self.day_embed(day)
                day_enc = day_enc.view(B, T_tube, tubelet_size, -1).mean(dim=2)
                enc = enc + day_enc

        enc = enc.unsqueeze(2).expand(-1, -1, n_spatial_tokens, -1) # [B, T_tube, S, D]
        enc = rearrange(enc, "b t s d -> b (t s) d")               # [B, N_tokens, D]
        return enc
