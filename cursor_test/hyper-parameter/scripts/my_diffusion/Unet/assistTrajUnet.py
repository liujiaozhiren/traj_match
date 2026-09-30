import torch
import torch.nn as nn

import cfg
from my_diffusion.Unet.base import AttnBlock, Downsample, ResnetBlock, Upsample, Normalize, get_timestep_embedding, nonlinearity



class AssistTrajUnetModel(nn.Module):  # 50 - 8 | 42   4 | 21  2 | 10  1 | 5
    """
        - input 分支: 长度 8 -> 4 -> 2 (ch_mult_input=(1,2,2)), num_res_blocks=2
        - info  分支: 长度 42 -> 21 -> 11 -> 6 -> 3 -> 2 (ch_mult_info=(1,2,2,2,2,2))
        - 两路在 L=2 汇合 (通道匹配后直接相加)
        - 仅用 input 的 up 解码: 2 -> 4 -> 8
        - Improved-UNet 规则：每个 up-level 有 num_res_blocks+1 个 ResBlock，
          最后一个 block 拼接“上一分辨率”的 skip（因此在 L=4 的 level 最后一次拼接通道是 256+128=384）
        """
    def __init__(self,
                 ch=128,
                 out_ch=2,
                 ch_mult_input=(1, 2, 2),  # 8 -> 4 -> 2
                 ch_mult_info=(1, 2, 2, 2, 2, 2),  # 42 -> 21 -> 11 -> 6 -> 3 -> 2
                 num_res_blocks=2,
                 attn_resolutions=(4, 2),
                 dropout=0.1,
                 in_channels=2,
                 resolution_input=8,
                 resolution_info=42,
                 resamp_with_conv=True):
        super().__init__()

        self.ch = ch
        self.out_ch = out_ch
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = set(attn_resolutions)
        self.temb_ch = ch * 4

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # ---------------- input encoder ----------------
        self.ch_mult_input = tuple(ch_mult_input)
        self.in_ch_mult_input = (1,) + self.ch_mult_input
        self.num_resolutions_input = len(self.ch_mult_input)
        self.conv_in_input = nn.Conv1d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        self.down_input = nn.ModuleList()
        curr_res = resolution_input  # 8
        block_in = ch
        for i_level in range(self.num_resolutions_input):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * self.ch_mult_input[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in, block_out, self.temb_ch, dropout))
                block_in = block_out
                # 注意力模块与 block 对齐，实际是否执行看 forward 时的长度
                attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions_input - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = (curr_res + 1) // 2  # ceil
            self.down_input.append(down)

        # input bottleneck channels
        bottleneck_c = block_in  # ch * ch_mult_input[-1] -> 256

        # ---------------- info encoder ----------------
        self.ch_mult_info = tuple(ch_mult_info)
        self.in_ch_mult_info = (1,) + self.ch_mult_info
        self.num_resolutions_info = len(self.ch_mult_info)
        self.conv_in_info = nn.Conv1d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        self.down_info = nn.ModuleList()
        curr_res = resolution_info  # 42
        block_in_info = ch
        for i_level in range(self.num_resolutions_info):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * self.ch_mult_info[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(block_in_info, block_out, self.temb_ch, dropout))
                block_in_info = block_out
                attn.append(AttnBlock(block_in_info))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions_info - 1:
                down.downsample = Downsample(block_in_info, resamp_with_conv)
                curr_res = (curr_res + 1) // 2
            self.down_info.append(down)

        # info bottleneck channels expected == bottleneck_c
        # 若不同，可用1x1对齐（这里两者都是 256，无需对齐）
        self.info_align = nn.Identity() if block_in_info == bottleneck_c else nn.Conv1d(block_in_info, bottleneck_c, 1)

        # ---------------- middle (after sum) ----------------
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(bottleneck_c, bottleneck_c, self.temb_ch, dropout)
        self.mid.attn_1 = AttnBlock(bottleneck_c)
        self.mid.block_2 = ResnetBlock(bottleneck_c, bottleneck_c, self.temb_ch, dropout)

        # ---------------- input decoder (Improved-UNet) ----------------
        self.up_input = nn.ModuleList()
        block_in_up = bottleneck_c  # start from bottleneck (256)
        for i_level in reversed(range(self.num_resolutions_input)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * self.ch_mult_input[i_level]
            # 前 num_res_blocks: 拼接当前分辨率 skip；最后 1 个：拼接上一分辨率 skip
            for i_block in range(num_res_blocks + 1):
                if i_block == num_res_blocks:
                    skip_c = ch * self.in_ch_mult_input[i_level]  # 上一分辨率
                else:
                    skip_c = ch * self.ch_mult_input[i_level]  # 当前分辨率
                in_c = block_in_up + skip_c
                block.append(ResnetBlock(in_c, block_out, self.temb_ch, dropout))
                attn.append(AttnBlock(block_out))
                block_in_up = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            up.upsample = Upsample(block_in_up, resamp_with_conv) if i_level != 0 else None
            self.up_input.append(up)

        # end
        self.norm_out = Normalize(block_in_up)  # = ch * ch_mult_input[0] = 128
        self.conv_out = nn.Conv1d(block_in_up, out_ch, kernel_size=3, stride=1, padding=1)

    # ---------------- forward ----------------
    def forward(self, x, info, t, extra_embed=None):
        """
        x:    (B, 2, 8)
        info: (B, 2, 42)
        t:    (B,)
        """
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra_embed is not None:
            temb = temb + extra_embed

        # ---------- input DOWN ----------
        hs_input = []
        h = self.conv_in_input(x)  # (B,128,8)
        hs_input.append(h)
        for i_level in range(self.num_resolutions_input):
            down = self.down_input[i_level]
            for i_block in range(self.num_res_blocks):
                h = down.block[i_block](h, temb)
                if h.size(-1) in self.attn_resolutions:  # 按长度触发注意力
                    h = down.attn[i_block](h)
                hs_input.append(h)
            if hasattr(down, 'downsample'):
                h = down.downsample(h)
                hs_input.append(h)
        h_input = h  # (B,256,2)

        # ---------- info DOWN ----------
        hs_info = []
        h = self.conv_in_info(info)  # (B,128,42)
        hs_info.append(h)
        for i_level in range(self.num_resolutions_info):
            down = self.down_info[i_level]
            for i_block in range(self.num_res_blocks):
                h = down.block[i_block](h, temb)
                if h.size(-1) in self.attn_resolutions:
                    h = down.attn[i_block](h)
                hs_info.append(h)
            if hasattr(down, 'downsample'):
                h = down.downsample(h)
                hs_info.append(h)
        h_info = self.info_align(h)  # -> (B,256,2)

        # ---------- middle (sum & process at L=2) ----------
        h = h_input + h_info
        h = self.mid.block_1(h, temb)
        if h.size(-1) in self.attn_resolutions:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # ---------- input UP ----------
        up_idx = 0
        for i_level in reversed(range(self.num_resolutions_input)):
            up = self.up_input[up_idx]
            for i_block in range(self.num_res_blocks + 1):
                ht = hs_input.pop()
                # 长度对齐（只 pad 主分支）
                if ht.size(-1) != h.size(-1):
                    h = F.pad(h, (0, ht.size(-1) - h.size(-1)))
                h = up.block[i_block](torch.cat([h, ht], dim=1), temb)
                if h.size(-1) in self.attn_resolutions:
                    h = up.attn[i_block](h)
            if up.upsample is not None:
                h = up.upsample(h)
            up_idx += 1

        # ---------- END ----------
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)  # (B, out_ch=2, 8)
        return h

class AssistTrajUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = cfg.dim
        self.unet = AssistTrajUnetModel()

    def forward(self, x, info, t):

        return self.unet(x, info, t)