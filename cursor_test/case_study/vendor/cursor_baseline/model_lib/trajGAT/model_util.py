import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import numpy as np

def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field]).sum(-1, keepdim=True)}

    return func


def scaled_exp(field, scale_constant):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field] / scale_constant).clamp(-5, 5))}

    return func


def message_func(edges):
    # message UDF for equation (3) & (4)
    return {"V_h": edges.src["V_h"], "score": edges.data["score"]}


def reduce_func(nodes):
    # reduce UDF for equation (3) & (4)
    # equation (3)
    alpha = F.softmax(nodes.mailbox["score"], dim=1)
    # equation (4)
    h = torch.sum(alpha * nodes.mailbox["V_h"], dim=1)
    return {"V_h": h}

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, use_bias=False):
        super(MultiHeadAttention,self).__init__()
        assert d_model % n_head == 0, "d_model must can be divisible by n_head"
        self.dim_head = d_model // n_head
        self.n_head = n_head

        self.W_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.W_v = nn.Linear(d_model, d_model, bias=use_bias)

    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst("K_h", "Q_h", "score"))
        g.apply_edges(scaled_exp("score", np.sqrt(self.dim_head)))

        g.update_all(message_func, reduce_func)

    def forward(self, g, h):
        Q_h = self.W_q(h).view(-1, self.n_head, self.dim_head)
        K_h = self.W_k(h).view(-1, self.n_head, self.dim_head)
        V_h = self.W_v(h).view(-1, self.n_head, self.dim_head)

        # Reshaping into [num_nodes, num_heads, feat_dim] to
        # get projections for multi-head attention
        g.ndata["Q_h"] = Q_h
        g.ndata["K_h"] = K_h
        g.ndata["V_h"] = V_h

        self.propagate_attention(g)

        head_out = g.ndata["V_h"]

        return head_out #【node num, n_head, d_head】


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout, layer_norm, batch_norm):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.W_o = nn.Linear(d_model, d_model)

        if batch_norm:
            self.norm_part1 = nn.BatchNorm1d(d_model)
        elif layer_norm:
            self.norm_part1 = nn.LayerNorm(d_model)

        # FFN
        self.FFN_layer1 = nn.Linear(d_model, d_model * 2)
        self.FFN_layer2 = nn.Linear(d_model * 2, d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if batch_norm:
            self.norm_part2 = nn.BatchNorm1d(d_model)
        elif layer_norm:
            self.norm_part2 = nn.LayerNorm(d_model)

    def forward(self, g, h):
        h_in1 = h  # for first residual connection

        # multi-head attention out
        attn_out = self.attention(g, h)
        h = attn_out.view(-1, self.d_model)

        h = self.dropout1(h)
        h = self.W_o(h)
        h = self.norm_part1(h_in1 + h)  # residual connection & normalization

        h_in2 = h  # for second residual connection

        # FFN
        h = self.FFN_layer1(h)
        h = F.relu(h)
        h = self.dropout2(h)
        h = self.FFN_layer2(h)

        h = self.norm_part2(h + h_in2)

        return h


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super(Encoder, self).__init__()
        self.encoder = _get_clones(encoder_layer, num_layers)

    def forward(self, g: dgl.DGLGraph, h):
        for layer in self.encoder:
            h = layer(g, h)
        g.ndata["h"] = h

        vectors = dgl.readout_nodes(g, "h", op="mean")

        # # 只选取 真实节点 进行mean
        # all_feat = g.ndata["h"]
        # all_flag = g.ndata["flag"]
        # print(all_flag[:600])
        # vectors = None

        return vectors  # [graph num, d_model]


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])