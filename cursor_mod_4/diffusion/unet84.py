from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import cfg
from my_diffusion.unet.base import (
    AttnBlock,
    Downsample,
    Normalize,
    ResnetBlock,
    Upsample,
    get_timestep_embedding,
    nonlinearity,
)


class AssistTrajUnetModel_8_4(nn.Module):
    """
    Two-branch UNet for (info_len=8, x_len=4).

    - x branch: 4 -> 2, then decode 2 -> 4
    - info branch: 8 -> 4 -> 2
    - merge at L=2 by sum, decode using x-branch skips
    """

    def __init__(
        self,
        ch: int = cfg.dim,
        out_ch: int = 2,
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        in_channels: int = 2,
        attn_resolutions: tuple[int, ...] = (4, 2),
        resamp_with_conv: bool = True,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = ch * 4
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = set(attn_resolutions)

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([nn.Linear(ch, self.temb_ch), nn.Linear(self.temb_ch, self.temb_ch)])

        # ---- x encoder (len 4 -> 2) ----
        self.conv_in_x = nn.Conv1d(in_channels, ch, 3, 1, 1)
        self.x_block0 = nn.ModuleList([ResnetBlock(ch, ch, False, dropout, self.temb_ch) for _ in range(num_res_blocks)])
        self.x_attn0 = nn.ModuleList([AttnBlock(ch) for _ in range(num_res_blocks)])
        self.x_down = Downsample(ch, resamp_with_conv)  # 4 -> 2
        ch2 = ch * 2
        self.x_block1 = nn.ModuleList([ResnetBlock(ch, ch2, False, dropout, self.temb_ch)] + [
            ResnetBlock(ch2, ch2, False, dropout, self.temb_ch) for _ in range(num_res_blocks - 1)
        ])
        self.x_attn1 = nn.ModuleList([AttnBlock(ch2) for _ in range(num_res_blocks)])

        # ---- info encoder (len 8 -> 4 -> 2) ----
        self.conv_in_info = nn.Conv1d(in_channels, ch, 3, 1, 1)
        self.i_block0 = nn.ModuleList([ResnetBlock(ch, ch, False, dropout, self.temb_ch) for _ in range(num_res_blocks)])
        self.i_attn0 = nn.ModuleList([AttnBlock(ch) for _ in range(num_res_blocks)])
        self.i_down0 = Downsample(ch, resamp_with_conv)  # 8 -> 4
        self.i_block1 = nn.ModuleList([ResnetBlock(ch, ch2, False, dropout, self.temb_ch)] + [
            ResnetBlock(ch2, ch2, False, dropout, self.temb_ch) for _ in range(num_res_blocks - 1)
        ])
        self.i_attn1 = nn.ModuleList([AttnBlock(ch2) for _ in range(num_res_blocks)])
        self.i_down1 = Downsample(ch2, resamp_with_conv)  # 4 -> 2
        self.i_block2 = nn.ModuleList([ResnetBlock(ch2, ch2, False, dropout, self.temb_ch) for _ in range(num_res_blocks)])
        self.i_attn2 = nn.ModuleList([AttnBlock(ch2) for _ in range(num_res_blocks)])

        # ---- middle (len 2) ----
        self.mid_block1 = ResnetBlock(ch2, ch2, False, dropout, self.temb_ch)
        self.mid_attn = AttnBlock(ch2)
        self.mid_block2 = ResnetBlock(ch2, ch2, False, dropout, self.temb_ch)

        # ---- x decoder (len 2 -> 4) ----
        # use skips: from x_block1 outputs and pre-downsample outputs
        self.up_block1 = nn.ModuleList(
            [
                ResnetBlock(ch2 + ch2, ch2, False, dropout, self.temb_ch),
                ResnetBlock(ch2 + ch2, ch2, False, dropout, self.temb_ch),
            ]
        )
        self.up_attn1 = nn.ModuleList([AttnBlock(ch2) for _ in range(2)])
        self.up = Upsample(ch2, resamp_with_conv)  # 2 -> 4
        self.up_block0 = nn.ModuleList(
            [
                ResnetBlock(ch2 + ch, ch, False, dropout, self.temb_ch),
                ResnetBlock(ch + ch, ch, False, dropout, self.temb_ch),
            ]
        )
        self.up_attn0 = nn.ModuleList([AttnBlock(ch) for _ in range(2)])

        self.norm_out = Normalize(ch)
        self.conv_out = nn.Conv1d(ch, out_ch, 3, 1, 1)

    def forward(self, x: torch.Tensor, info: torch.Tensor, t: torch.Tensor, extra=None) -> torch.Tensor:
        # temb
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra is not None:
            temb = temb + extra

        # x down
        hs = []
        h = self.conv_in_x(x)
        hs.append(h)  # skip at 4
        for blk, attn in zip(self.x_block0, self.x_attn0):
            h = blk(h, temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)
            hs.append(h)
        h = self.x_down(h)  # 2
        for blk, attn in zip(self.x_block1, self.x_attn1):
            h = blk(h, temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)
            hs.append(h)
        h_x = h

        # info down
        h = self.conv_in_info(info)
        for blk, attn in zip(self.i_block0, self.i_attn0):
            h = blk(h, temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)
        h = self.i_down0(h)  # 4
        for blk, attn in zip(self.i_block1, self.i_attn1):
            h = blk(h, temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)
        h = self.i_down1(h)  # 2
        for blk, attn in zip(self.i_block2, self.i_attn2):
            h = blk(h, temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)
        h_info = h

        # middle merge
        h = h_x + h_info
        h = self.mid_block1(h, temb)
        if h.size(-1) in self.attn_resolutions:
            h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        # up at len2 with two skips from hs (last two are len2)
        for blk, attn in zip(self.up_block1, self.up_attn1):
            ht = hs.pop()
            if ht.size(-1) != h.size(-1):
                diff = ht.size(-1) - h.size(-1)
                if diff > 0:
                    h = F.pad(h, (0, diff))
                else:
                    ht = F.pad(ht, (0, -diff))
            h = blk(torch.cat([h, ht], dim=1), temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)

        h = self.up(h)  # 4

        # up at len4 with two skips from hs (len4)
        for blk, attn in zip(self.up_block0, self.up_attn0):
            ht = hs.pop()
            if ht.size(-1) != h.size(-1):
                diff = ht.size(-1) - h.size(-1)
                if diff > 0:
                    h = F.pad(h, (0, diff))
                else:
                    ht = F.pad(ht, (0, -diff))
            h = blk(torch.cat([h, ht], dim=1), temb)
            if h.size(-1) in self.attn_resolutions:
                h = attn(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        return self.conv_out(h)


class AssistTrajUnet84(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = AssistTrajUnetModel_8_4()

    def forward(self, xt: torch.Tensor, info: torch.Tensor, t: torch.Tensor, extra=None):
        return self.unet(xt, info, t, extra=extra)

