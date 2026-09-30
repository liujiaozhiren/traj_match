import torch
import torch.nn as nn
import torch.nn.functional as F

import cfg
from my_diffusion.unet.base import (
    AttnBlock, Downsample, ResnetBlock, Upsample, Normalize,
    get_timestep_embedding, nonlinearity
)


class AssistTrajUnetPivotModel(nn.Module):
    """
    - x 分支: 仅在 L=1 上做若干 ResBlock（不下采样）
    - info 分支: 42 -> 21 -> 11 -> 6 -> 3 -> 2 -> 1
    - L=1 汇合后 middle，再直接输出 (B,2,1)
    """
    def __init__(self,
                 ch=128,
                 out_ch=2,
                 # x 分支只有一个分辨率（L=1），做 num_res_blocks 个 ResBlock 即可
                 num_res_blocks_x=2,
                 # info 分支连续下采样至 L=1
                 ch_mult_info=(1, 2, 2, 2, 2, 2, 2),  # 42->21->11->6->3->2->1
                 num_res_blocks_info=2,
                 attn_resolutions=(3, 1),  # 在 L=3 与 L=1 触发注意力
                 dropout=0.1,
                 in_channels=2,
                 resolution_x=1,
                 resolution_info=42,
                 resamp_with_conv=True):
        super().__init__()

        self.ch = ch
        self.out_ch = out_ch
        self.temb_ch = ch * 4
        self.attn_resolutions = set(attn_resolutions)

        # ---- timestep embedding ----
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # ---------------- x encoder @ L=1 ----------------
        self.conv_in_x = nn.Conv1d(in_channels, ch, kernel_size=3, stride=1, padding=1)
        self.down_x = nn.Module()
        self.down_x.block = nn.ModuleList()
        self.down_x.attn = nn.ModuleList()
        block_in_x = ch
        for _ in range(num_res_blocks_x):
            self.down_x.block.append(
                ResnetBlock(block_in_x, ch, self.temb_ch, dropout)
            )
            self.down_x.attn.append(AttnBlock(ch))
            block_in_x = ch  # 保持通道不变（128）

        bottleneck_c = block_in_x  # 128 起，若你想与原 256 对齐，可改 ch_mult

        # ---------------- info encoder: 42 -> ... -> 1 ----------------
        self.ch_mult_info = tuple(ch_mult_info)
        self.num_resolutions_info = len(self.ch_mult_info)
        self.conv_in_info = nn.Conv1d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        self.down_info = nn.ModuleList()
        curr_res = resolution_info  # 42
        block_in_info = ch
        for i_level in range(self.num_resolutions_info):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * self.ch_mult_info[i_level]
            for _ in range(num_res_blocks_info):
                block.append(ResnetBlock(block_in_info, block_out, self.temb_ch, dropout))
                block_in_info = block_out
                attn.append(AttnBlock(block_in_info))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions_info - 1:
                down.downsample = Downsample(block_in_info, resamp_with_conv)
                curr_res = (curr_res + 1) // 2  # ceil 下采样
            self.down_info.append(down)

        # 对齐 info bottleneck 到 x bottleneck 的通道数
        self.info_align = nn.Identity() if block_in_info == bottleneck_c else nn.Conv1d(block_in_info, bottleneck_c, 1)

        # ---------------- middle @ L=1 ----------------
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(bottleneck_c, bottleneck_c, self.temb_ch, dropout)
        self.mid.attn_1   = AttnBlock(bottleneck_c)
        self.mid.block_2 = ResnetBlock(bottleneck_c, bottleneck_c, self.temb_ch, dropout)

        # ---------------- end ----------------
        self.norm_out = Normalize(bottleneck_c)
        self.conv_out = nn.Conv1d(bottleneck_c, out_ch, kernel_size=3, stride=1, padding=1)

        # 保存配置（可选）
        self.num_res_blocks_x = num_res_blocks_x
        self.num_res_blocks_info = num_res_blocks_info

    def forward(self, x, info, t, extra_embed=None):
        """
        x    : (B, 2, 1)
        info : (B, 2, 42)
        t    : (B,)
        """
        # --- timestep embedding ---
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra_embed is not None:
            temb = temb + extra_embed

        # --- x branch @ L=1 ---
        hx = self.conv_in_x(x)  # (B,128,1)
        for i_block in range(self.num_res_blocks_x):
            hx = self.down_x.block[i_block](hx, temb)
            if hx.size(-1) in self.attn_resolutions:  # L=1 时触发
                hx = self.down_x.attn[i_block](hx)

        # --- info branch: 42 -> ... -> 1 ---
        h = self.conv_in_info(info)  # (B,128,42)
        for i_level in range(self.num_resolutions_info):
            down = self.down_info[i_level]
            for i_block in range(self.num_res_blocks_info):
                h = down.block[i_block](h, temb)
                if h.size(-1) in self.attn_resolutions:
                    h = down.attn[i_block](h)
            if hasattr(down, 'downsample'):
                h = down.downsample(h)
        h_info = self.info_align(h)  # -> (B, bottleneck_c, 1)

        # --- fuse @ L=1 ---
        h = hx + h_info
        h = self.mid.block_1(h, temb)
        if h.size(-1) in self.attn_resolutions:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # --- out ---
        h = self.norm_out(h)
        h = nonlinearity(h)
        y = self.conv_out(h)  # (B, 2, 1)
        return y


class AssistTrajUnetPivot(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = cfg.dim
        self.unet = AssistTrajUnetPivotModel()

    def forward(self, x, info, t):
        return self.unet(x, info, t)