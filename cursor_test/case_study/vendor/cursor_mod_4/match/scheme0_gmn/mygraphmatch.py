# Migrated from match_model/my/mygraphmatch.py (scheme0 exact behavior)
import math

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F

import cfg


class GraphEncoderDGL(nn.Module):
    def __init__(self, node_in, edge_in=None, node_hidden_sizes=(128,), edge_hidden_sizes=None):
        super().__init__()
        self.node_mlp = self._mlp(node_in, list(node_hidden_sizes))
        self.edge_mlp = None
        if edge_in is not None and edge_hidden_sizes is not None:
            self.edge_mlp = self._mlp(edge_in, list(edge_hidden_sizes))

    @staticmethod
    def _mlp(in_dim, sizes):
        layers = []
        last = in_dim
        for h in sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        if len(layers) >= 2 and isinstance(layers[-1], nn.ReLU):
            layers = layers[:-1]
        return nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, g: dgl.DGLGraph):
        x = g.ndata["feat"]
        node_states = self.node_mlp(x)
        edge_states = None
        if self.edge_mlp is not None and "ef" in g.edata:
            edge_states = self.edge_mlp(g.edata["ef"])
        return node_states, edge_states


class GraphPropLayerDGL(nn.Module):
    def __init__(
        self,
        node_state_dim,
        edge_state_dim=0,
        edge_hidden_sizes=(128,),
        node_hidden_sizes=(128,),
        node_update_type="residual",
        use_reverse_direction=True,
        reverse_dir_param_different=True,
        layer_norm=False,
        prop_type="embedding",
    ):
        super().__init__()
        self.Dn = node_state_dim
        self.De = edge_state_dim
        self.use_rev = use_reverse_direction
        self.rev_diff = reverse_dir_param_different
        self.layer_norm = layer_norm
        self.prop_type = prop_type

        in_m = node_state_dim * 2 + (edge_state_dim if edge_state_dim > 0 else 0)
        self.msg_mlp = self._mlp(in_m, list(edge_hidden_sizes))
        if self.use_rev and self.rev_diff:
            self.msg_mlp_rev = self._mlp(in_m, list(edge_hidden_sizes))
        else:
            self.msg_mlp_rev = self.msg_mlp

        self.node_update_type = node_update_type
        if node_update_type == "gru":
            in_u = self.Dn * 2 if prop_type == "embedding" else self.Dn * 3
            self.gru = nn.GRUCell(in_u, self.Dn)
        else:
            in_u = self.Dn * 2 if prop_type == "embedding" else self.Dn * 3
            self.up_mlp = self._mlp(in_u, list(node_hidden_sizes) + [self.Dn])

        if self.layer_norm:
            self.ln_msg = nn.LayerNorm(node_state_dim)
            self.ln_upd = nn.LayerNorm(node_state_dim)

    @staticmethod
    def _mlp(in_dim, sizes):
        layers, last = [], in_dim
        for h in sizes:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        if len(layers) >= 2 and isinstance(layers[-1], nn.ReLU):
            layers = layers[:-1]
        return nn.Sequential(*layers)

    def _prop_once(self, g: dgl.DGLGraph, node_states, edge_states, mlp):
        g = g.local_var()
        g.ndata["h"] = node_states
        if edge_states is not None:
            g.edata["ef_"] = edge_states

        def edge_udf(edges):
            h_src = edges.src["h"]
            h_dst = edges.dst["h"]
            if "ef_" in edges.data:
                z = torch.cat([h_src, h_dst, edges.data["ef_"]], dim=-1)
            else:
                z = torch.cat([h_src, h_dst], dim=-1)
            m = mlp(z)
            return {"m": m}

        g.apply_edges(edge_udf)
        g.update_all(fn.copy_e("m", "m"), fn.sum("m", "agg"))
        agg = g.ndata.pop("agg")
        if agg.shape[-1] != self.Dn:
            proj = getattr(self, "_proj_msg", None)
            if proj is None:
                self._proj_msg = nn.Linear(agg.shape[-1], self.Dn).to(agg.device)
                proj = self._proj_msg
            agg = proj(agg)
        return agg

    def _aggregate_messages(self, g, node_states, edge_states):
        agg = self._prop_once(g, node_states, edge_states, self.msg_mlp)
        if self.use_rev:
            grev = dgl.reverse(g, share_ndata=True, share_edata=True)
            agg_rev = self._prop_once(grev, node_states, edge_states, self.msg_mlp_rev)
            agg = agg + agg_rev
        if self.layer_norm:
            agg = self.ln_msg(agg)
        return agg

    def _node_update(self, old_h, inputs_cat):
        if self.node_update_type == "gru":
            return self.gru(inputs_cat, old_h)
        upd = self.up_mlp(inputs_cat)
        if self.layer_norm:
            upd = self.ln_upd(upd)
        if self.node_update_type == "mlp":
            return upd
        return old_h + upd

    def forward(self, g: dgl.DGLGraph, node_states, edge_states=None, extra_inputs=None):
        agg = self._aggregate_messages(g, node_states, edge_states)
        if self.prop_type == "embedding":
            inputs_cat = torch.cat([agg, node_states], dim=-1)
        else:
            if extra_inputs is None:
                raise ValueError("matching 模式需要提供 attention_input")
            inputs_cat = torch.cat([agg, extra_inputs, node_states], dim=-1)
        return self._node_update(node_states, inputs_cat)


def _pairwise_dot(x, y):
    return x @ y.t()


def _pairwise_euclidean(x, y):
    s = 2 * (x @ y.t())
    s -= x.pow(2).sum(dim=1, keepdim=True)
    s -= y.pow(2).sum(dim=1, keepdim=True).t()
    return s


def _pairwise_cosine(x, y, eps=1e-12):
    x = x / (x.norm(dim=-1, keepdim=True) + eps)
    y = y / (y.norm(dim=-1, keepdim=True) + eps)
    return x @ y.t()


def compute_cross_attention_pair(x, y, sim="dotproduct"):
    if sim == "dotproduct":
        a = _pairwise_dot(x, y)
    elif sim == "euclidean":
        a = _pairwise_euclidean(x, y)
    elif sim == "cosine":
        a = _pairwise_cosine(x, y)
    else:
        raise ValueError(f"unknown sim: {sim}")
    a_x = torch.softmax(a, dim=1)
    a_y = torch.softmax(a, dim=0)
    attn_x = a_x @ y
    attn_y = a_y.t() @ x
    return attn_x, attn_y


class GraphPropMatchingLayerDGL(GraphPropLayerDGL):
    def __init__(self, *args, similarity="dotproduct", **kwargs):
        super().__init__(*args, **kwargs)
        self.similarity = similarity

    @staticmethod
    def _get_batch_slices(g: dgl.DGLGraph):
        if hasattr(g, "batch_num_nodes"):
            nn_list = g.batch_num_nodes().tolist()
        else:
            nn_list = [gi.num_nodes() for gi in dgl.unbatch(g)]
        offs, cur = [], 0
        for n in nn_list:
            offs.append((cur, cur + n))
            cur += n
        return offs

    def forward(self, g: dgl.DGLGraph, node_states, edge_states=None, extra_inputs=None):
        agg = self._aggregate_messages(g, node_states, edge_states)
        slices = self._get_batch_slices(g)
        attn_all = torch.zeros_like(node_states)
        if cfg.graph_between_attention:
            num_graphs = len(slices)
            if num_graphs % 2 != 0:
                raise ValueError(f"Batched 图的子图个数必须为偶数，现在是 {num_graphs}")
            for i in range(0, num_graphs, 2):
                s1, e1 = slices[i]
                s2, e2 = slices[i + 1]
                x = node_states[s1:e1]
                y = node_states[s2:e2]
                ax, ay = compute_cross_attention_pair(x, y, sim=self.similarity)
                attn_all[s1:e1] = ax
                attn_all[s2:e2] = ay
        else:
            for i in range(len(slices)):
                s, e = slices[i]
                x = node_states[s:e]
                ax, _ = compute_cross_attention_pair(x, x, sim=self.similarity)
                attn_all[s:e] = ax
        attention_input = node_states - attn_all
        inputs_cat = torch.cat([agg, attention_input, node_states], dim=-1)
        return self._node_update(node_states, inputs_cat)


class GraphAggregatorDGL(nn.Module):
    def __init__(self, node_state_dim, node_hidden_sizes=(128,), graph_transform_sizes=None, gated=True, readout="sum"):
        super().__init__()
        self.gated = gated
        self.readout = readout
        out_node = node_hidden_sizes[-1]
        if gated:
            out_node = node_hidden_sizes[-1] * 2
        self.node_mlp = GraphEncoderDGL._mlp(node_state_dim, list(node_hidden_sizes[:-1]) + [out_node])
        self.post_mlp = None
        if graph_transform_sizes and len(graph_transform_sizes) > 0:
            self.post_mlp = GraphEncoderDGL._mlp(node_hidden_sizes[-1], list(graph_transform_sizes))

    def forward(self, g: dgl.DGLGraph, node_states):
        g = g.local_var()
        z = self.node_mlp(node_states)
        if self.gated:
            H = z.shape[-1] // 2
            gate = torch.sigmoid(z[:, :H])
            z = z[:, H:] * gate
        g.ndata["z"] = z
        if self.readout == "sum":
            gs = dgl.readout_nodes(g, "z", op="sum") if hasattr(dgl, "readout_nodes") else dgl.sum_nodes(g, "z")
        elif self.readout == "mean":
            gs = dgl.mean_nodes(g, "z")
        elif self.readout == "max":
            gs = dgl.max_nodes(g, "z")
        else:
            raise ValueError("unknown readout")
        if self.post_mlp is not None:
            gs = self.post_mlp(gs)
        return gs


class GraphEmbeddingNetDGL(nn.Module):
    def __init__(
        self,
        encoder: GraphEncoderDGL,
        aggregator: GraphAggregatorDGL,
        node_state_dim,
        edge_state_dim=0,
        edge_hidden_sizes=(128,),
        node_hidden_sizes=(128,),
        n_prop_layers=3,
        share_prop_params=False,
        node_update_type="residual",
        use_reverse_direction=True,
        reverse_dir_param_different=True,
        layer_norm=False,
        layer_class=GraphPropLayerDGL,
        prop_type="embedding",
    ):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.layers = nn.ModuleList()
        for i in range(n_prop_layers):
            if i > 0 and share_prop_params:
                self.layers.append(self.layers[0])
            else:
                self.layers.append(
                    layer_class(
                        node_state_dim,
                        edge_state_dim,
                        edge_hidden_sizes=edge_hidden_sizes,
                        node_hidden_sizes=node_hidden_sizes,
                        node_update_type=node_update_type,
                        use_reverse_direction=use_reverse_direction,
                        reverse_dir_param_different=reverse_dir_param_different,
                        layer_norm=layer_norm,
                        prop_type=prop_type,
                    )
                )

    def forward(self, g: dgl.DGLGraph):
        node_states, edge_states = self.encoder(g)
        for layer in self.layers:
            node_states = layer(g, node_states, edge_states=edge_states)
        return self.aggregator(g, node_states)


class GraphMatchingNetDGL(GraphEmbeddingNetDGL):
    def __init__(self, similarity="dotproduct", **kwargs):
        super().__init__(
            layer_class=lambda *a, **k: GraphPropMatchingLayerDGL(*a, similarity=similarity, **k),
            prop_type="matching",
            **kwargs,
        )

