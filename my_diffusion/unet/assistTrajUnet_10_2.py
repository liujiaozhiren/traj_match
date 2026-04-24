import torch
import torch.nn as nn

import cfg
from my_diffusion.unet.base import AttnBlock, Downsample, ResnetBlock, Upsample, Normalize, get_timestep_embedding, nonlinearity

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

# 依赖：ResnetBlock / AttnBlock / Downsample / Upsample / Normalize / nonlinearity / get_timestep_embedding
# 来自你项目的 base.py 等，保持不变

class AssistTrajUnetModel_10_2(nn.Module):
    """
        目标：input 长度=2（2->2->2），info 长度=10（10->5->3->2->2），在 L=2 汇合，仅用 input 分支解码（2->2->2），输出 (B,2,2)
    """
    def __init__(self,
                 ch=cfg.dim,
                 out_ch=2,
                 # 注意：这里 input=2 的分辨率序列；info=10 的分辨率序列
                 res_list_input=(2, 2, 2),
                 res_list_info=(10, 5, 3, 2, 2),
                 ch_mult_input=(1, 2, 2),          # 你可按需调倍率；不影响长度
                 ch_mult_info=(1, 2, 2, 2, 2),
                 num_res_blocks=2,
                 attn_resolutions=(3, 2),
                 dropout=0.1,
                 in_channels=2,
                 resamp_with_conv=True):
        super().__init__()
        if ch is None:
            ch = cfg.dim

        assert len(res_list_input) == len(ch_mult_input)
        assert len(res_list_info)  == len(ch_mult_info)
        assert res_list_input[-1] == 2 and res_list_info[-1] == 2

        self.ch = ch
        self.out_ch = out_ch
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = set(attn_resolutions)
        self.temb_ch = ch * 4

        self.res_list_input = tuple(res_list_input)  # (2,2,2)
        self.res_list_info  = tuple(res_list_info)   # (10,5,3,2,2)

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.ch, self.temb_ch),
            nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # ---------------- input encoder (2->2->2) ----------------
        self.ch_mult_input = tuple(ch_mult_input)
        self.in_ch_mult_input = (1,) + self.ch_mult_input
        self.num_resolutions_input = len(self.ch_mult_input)
        self.conv_in_input = nn.Conv1d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        self.down_input = nn.ModuleList()
        block_in = ch
        for i_level in range(self.num_resolutions_input):
            cur_len = self.res_list_input[i_level]   # 全是2
            block = nn.ModuleList()
            attn  = nn.ModuleList()
            block_out = ch * self.ch_mult_input[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out, False, dropout, self.temb_ch))
                block_in = block_out
                attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn  = attn
            # 2->2 不下采样
            down.downsample = None
            self.down_input.append(down)

        bottleneck_c = block_in  # input 编码末端通道（L=2）

        # ---------------- info encoder (10->5->3->2->2) ----------------
        self.ch_mult_info = tuple(ch_mult_info)
        self.in_ch_mult_info = (1,) + self.ch_mult_info
        self.num_resolutions_info = len(self.ch_mult_info)
        self.conv_in_info = nn.Conv1d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        self.down_info = nn.ModuleList()
        block_in_info = ch
        for i_level in range(self.num_resolutions_info):
            cur_len = self.res_list_info[i_level]
            block = nn.ModuleList()
            attn  = nn.ModuleList()
            block_out = ch * self.ch_mult_info[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in_info, block_out, False, dropout, self.temb_ch))
                block_in_info = block_out
                attn.append(AttnBlock(block_in_info))
            down = nn.Module()
            down.block = block
            down.attn  = attn
            # 根据长度是否变小决定是否下采样
            if i_level != self.num_resolutions_info - 1:
                next_len = self.res_list_info[i_level + 1]
                down.downsample = Downsample(block_in_info, resamp_with_conv) if next_len < cur_len else None
            self.down_info.append(down)

        # 汇合前通道对齐（若不同，用1x1）
        self.info_align = nn.Identity() if block_in_info == bottleneck_c else nn.Conv1d(block_in_info, bottleneck_c, 1)

        # ---------------- middle (L=2) ----------------
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(bottleneck_c, bottleneck_c, False, dropout, self.temb_ch)
        self.mid.attn_1  = AttnBlock(bottleneck_c)
        self.mid.block_2 = ResnetBlock(bottleneck_c, bottleneck_c, False, dropout, self.temb_ch)

        # ---------------- “干跑”input-encoder，记录真实 hs_input push 通道栈 ----------------
        hs_ch = []
        cur_c = ch
        hs_ch.append(cur_c)  # conv_in_input 之后 push
        for i_level in range(self.num_resolutions_input):
            block_out = ch * self.ch_mult_input[i_level]
            for _ in range(self.num_res_blocks):
                cur_c = block_out
                hs_ch.append(cur_c)
            # input=2->2 无下采样，不再 push
        hs_ch_stack = hs_ch.copy()

        # ---------------- input decoder（2->2->2，无上采样） ----------------
        self.up_input = nn.ModuleList()
        block_in_up = bottleneck_c
        for i_level in reversed(range(self.num_resolutions_input)):
            block = nn.ModuleList()
            attn  = nn.ModuleList()
            block_out = ch * self.ch_mult_input[i_level]

            # 对于 2->2：没有“上一分辨率更大”的情况，所以 extra_skip=0
            blocks_this_level = self.num_res_blocks  # 只做 num_res_blocks 个拼接

            for _ in range(blocks_this_level):
                assert len(hs_ch_stack) > 0, "hs_ch_stack 为空，skip 记录与实际不匹配"
                skip_c = hs_ch_stack.pop()
                in_c = block_in_up + skip_c
                block.append(ResnetBlock(in_c, block_out, False, dropout, self.temb_ch))
                attn.append(AttnBlock(block_out))
                block_in_up = block_out

            up = nn.Module()
            up.block = block
            up.attn  = attn
            # 2->2 不上采样
            up.upsample = None
            self.up_input.append(up)

        self.norm_out = Normalize(block_in_up)
        self.conv_out = nn.Conv1d(block_in_up, out_ch, kernel_size=3, stride=1, padding=1)

    # ---------------- forward ----------------
    def forward(self, x, info, t, extra=None):
        """
        x:    (B, 2, 2)   # input 分支（短）
        info: (B, 2, 10)  # info 分支（长）
        t:    (B,)
        return: (B, 2, 2)
        """
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)
        if extra is not None:
            temb = temb + extra

        # ---------- input DOWN (2->2->2) ----------
        hs_input = []
        h = self.conv_in_input(x)  # (B, ch, 2)
        hs_input.append(h)
        for i_level in range(self.num_resolutions_input):
            down = self.down_input[i_level]
            for i_block in range(self.num_res_blocks):
                h = down.block[i_block](h, temb)
                if h.size(-1) in self.attn_resolutions:
                    h = down.attn[i_block](h)
                hs_input.append(h)
            # 无下采样
        h_input = h  # (B, bottleneck_c, 2)

        # ---------- info DOWN (10->5->3->2->2) ----------
        hs_info = []
        h = self.conv_in_info(info)  # (B, ch, 10)
        hs_info.append(h)
        for i_level in range(self.num_resolutions_info):
            down = self.down_info[i_level]
            for i_block in range(self.num_res_blocks):
                h = down.block[i_block](h, temb)
                if h.size(-1) in self.attn_resolutions:
                    h = down.attn[i_block](h)
                hs_info.append(h)
            if getattr(down, 'downsample', None) is not None:
                h = down.downsample(h)
                hs_info.append(h)
        h_info = self.info_align(h)  # -> (B, bottleneck_c, 2)

        # ---------- middle (sum at L=2) ----------
        h = h_input + h_info
        h = self.mid.block_1(h, temb)
        if h.size(-1) in self.attn_resolutions:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # ---------- input UP (2->2->2) ----------
        up_idx = 0
        for i_level in reversed(range(self.num_resolutions_input)):
            up = self.up_input[up_idx]
            for i_block in range(self.num_res_blocks):
                ht = hs_input.pop()
                # 长度对齐（理论上全是2，这里留兜底）
                if ht.size(-1) != h.size(-1):
                    diff = ht.size(-1) - h.size(-1)
                    if diff > 0:
                        h = F.pad(h, (0, diff))
                    else:
                        ht = F.pad(ht, (0, -diff))
                h = up.block[i_block](torch.cat([h, ht], dim=1), temb)
                if h.size(-1) in self.attn_resolutions:
                    h = up.attn[i_block](h)
            # 无上采样
            up_idx += 1

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)  # (B, 2, 2)
        return h