import copy

import dgl
import torch
from torch import nn

import cfg
from diffusion.predict.lstm import TrajPredHandler
from model.ST2Vec.model.model_network import ST_Encoder
from model.traj2simvec.model import Traj2SimVec
from model.trajGAT.model import GraphTransformer
from my_diffusion.fixLdiff.model import FixLenDiff


class ModelHandler(nn.Module):
    def __init__(self, dim=cfg.dim, model_name='traj2simvec', embedding=None, network=None):
        super(ModelHandler, self).__init__()
        self.model_name = model_name
        if model_name == 'traj2simvec':
            self.model_pre = Traj2SimVec(dim)
            self.model_post = Traj2SimVec(dim)
        elif model_name == 'trajGAT':
            embedding_pre = copy.deepcopy(embedding)
            embedding_post = copy.deepcopy(embedding)
            self.model_pre = GraphTransformer(embedding_pre)
            self.model_post = GraphTransformer(embedding_post)
        elif model_name == 'ST2Vec':
            self.pre_network = copy.deepcopy(network)
            self.post_network = copy.deepcopy(network)
            self.model_pre = ST_Encoder()
            self.model_post = ST_Encoder()
        elif model_name == 'pred_lstm':
            self.gen_pre = TrajPredHandler()
            self.gen_post = TrajPredHandler()
        elif model_name == 'mydiff':
            self.diff_pre = FixLenDiff()
            self.diff_post = FixLenDiff()
            self.match = None
        else:
            raise ValueError(f"Model {model_name} is not supported.")

    def forward(self, x):
        if self.model_name == 'traj2simvec':
            x0, x1 = x[0].to(cfg.device), x[1].to(cfg.device)
            out0 = self.model_pre(x0)[:, -1, :]
            out1 = self.model_post(x1)[:, -1, :]
            out = torch.cat([out0, out1], dim=0).view(2, -1, out0.shape[-1])
            return out
        elif self.model_name == 'trajGAT':
            batch_graphs_pre = dgl.batch(x[0]).to(cfg.device)  # (B*SAM, graph)
            batch_graphs_post = dgl.batch(x[1]).to(cfg.device)  # (B*SAM, graph)
            out_pre = self.model_pre(batch_graphs_pre)  # vecters [B*SAM, d_model]
            out_post = self.model_post(batch_graphs_post)  # vecters [B*SAM, d_model]
            out = torch.cat([out_pre, out_post], dim=0).view(2, -1, out_pre.shape[-1])
            return out
        elif self.model_name == 'ST2Vec':
            pre_ts_emb, pre_id_seq, post_ts_emb, post_id_seq = x
            out_pre = self.model_pre(self.pre_network, pre_id_seq, pre_ts_emb)
            out_post = self.model_post(self.post_network, post_id_seq, post_ts_emb)
            out = torch.cat([out_pre, out_post], dim=0).view(2, -1, out_pre.shape[-1])
            return out
        elif self.model_name == 'pred_lstm':
            len_tensor_pre, len_tensor_post = x[1][0].to(cfg.device), x[1][1].to(cfg.device)
            pre, post = x[0][0].to(cfg.device), x[0][1].to(cfg.device)
            pre0, pre1 = self.gen_pre(pre)
            post0, post1 = self.gen_post(post)
            return pre0, pre1, post0, post1, len_tensor_pre, len_tensor_post
        elif self.model_name == 'mydiff':
            pre, post = x[0].to(cfg.device), x[1].to(cfg.device)
            trajs_pre = torch.transpose(pre, 1, 2).to(cfg.device)
            trajs_post = torch.transpose(post, 1, 2).to(cfg.device)
            trajs_pre, trajs_post = trajs_pre[..., -42:], trajs_post[..., -42:]
            for i in range(cfg.diffusion_num):
                pre = self.diff_pre.infer_from_noise(trajs_pre)
                post = self.diff_post.infer_from_noise(trajs_post)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

    def gen_trajs(self, x):
        assert self.model_name == 'pred_lstm', "Only pred_lstm model supports trajectory generation."
        input, len_tensor, t = x
        b_s_l_d_pre = self.gen_pre.gen_trajs(input[0], len_tensor[0], t)
        b_s_l_d_post = self.gen_post.gen_trajs(input[1], len_tensor[1], t)
        return b_s_l_d_pre, b_s_l_d_post
