import torch
import cfg

def gen_attn_mask(L, n, device):
    attn_mask = torch.zeros((2 * L + 4 * n, 2 * L + 4 * n), dtype=torch.bool, device=device)
    sub_attn_mask = torch.ones((2 * n, 2 * n), dtype=torch.bool, device=device)
    for i in range(n):
        sub_attn_mask[2 * i:2 * i + 2, 2 * i:2 * i + 2] = False

    attn_mask[0:L, L:L + 2 * n] = True
    attn_mask[L:L + 2 * n, 0:L] = True
    attn_mask[L + 2 * n:2 * L + 2 * n, 2 * L + 2 * n:2 * L + 4 * n] = True
    attn_mask[2 * L + 2 * n:2 * L + 4 * n, L + 2 * n:2 * L + 2 * n] = True

    attn_mask[L:L + 2 * n, L:L + 2 * n] = sub_attn_mask
    attn_mask[2 * L + 2 * n:2 * L + 4 * n, 2 * L + 2 * n:2 * L + 4 * n] = sub_attn_mask
    return attn_mask


class AttnMatch(torch.nn.Module):
    def __init__(self, L, n, in_dim=2, dim=128, n2=cfg.hydra_tail):
        super(AttnMatch, self).__init__()
        self.L = L
        self.n = n
        self.n2 = n2
        self.attn_mask = gen_attn_mask(L, n, device='cpu')
        self.large_mask = gen_attn_mask(L, self.n2, device='cpu')
        # self.register_buffer('attn_mask', gen_attn_mask(L, n, device='cpu'))
        self.linear = torch.nn.Linear(in_dim, dim)
        layer = torch.nn.TransformerEncoderLayer(d_model=128, nhead=4, batch_first=True)
        self.attn1 = torch.nn.TransformerEncoder(layer, num_layers=3)
        # self.linear_out = torch.nn.Linear(dim * n * 4, 1)
        self.attn2 = torch.nn.TransformerEncoder(layer, num_layers=2)
        self.special_token = torch.nn.Parameter(torch.randn(1, 1, dim))

        # self.lstm = torch.nn.LSTM(input_size=dim, hidden_size=dim,
        #                           num_layers=2, batch_first=True, bidirectional=True)

        self.l2 = torch.nn.Linear(dim, 1)

    def forward(self, info_pre, info_post, hydra_pre, hydra_post):
        # info_pre, info_post: (B, L, 2)
        # hydra_pre, hydra_post: (B, n, 2, 2)+
        B = info_pre.size(0)
        x = torch.cat([info_pre, hydra_pre.view(B, self.n * 2, 2),
                       info_post, hydra_post.view(B, self.n * 2, 2)], dim=1)

        x = self.linear(x)  # (B, 2L+4n, dim)
        attn_mask = self.attn_mask.to(x.device)
        x_attn = self.attn1(x, mask=attn_mask)
        x_o1, x_o2 = x_attn[:, self.L:self.L + 2 * self.n], x_attn[:, 2 * self.L + 2 * self.n:2 * self.L + 4 * self.n]
        x_o = torch.cat([x_o1, x_o2], dim=1)
        # #
        # x_out1 = self.linear_out(x_o.view(B, -1))
        #
        sp = self.special_token.repeat(B, 1, 1)  # (B, 1, dim)
        x_int = torch.cat([sp, x_o], dim=1)  # (B, 1+4n, dim)
        x_out2 = self.attn2(x_int)[:, 0, :]
        #
        # x_out2, _ = self.lstm(x_o)
        # x_out2 = x_out2[:, -1, :]

        x_out2 = self.l2(x_out2)
        return x_out2

    def encode(self,  info_pre, info_post, hydra_pre, hydra_post):
        B = info_pre.size(0)
        x = torch.cat([info_pre, hydra_pre.view(B, self.n2 * 2, 2),
                       info_post, hydra_post.view(B, self.n2 * 2, 2)], dim=1)

        x = self.linear(x)  # (B, 2L+4n, dim)
        attn_mask = self.large_mask.to(x.device)
        x_attn = self.attn1(x, mask=attn_mask)
        x_o1, x_o2 = x_attn[:, self.L:self.L + 2 * self.n2], x_attn[:, 2 * self.L + 2 * self.n2:2 * self.L + 4 * self.n2]
        # B, 2n , dim
        return x_o1, x_o2