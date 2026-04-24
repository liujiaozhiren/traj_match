import torch
import torch.nn as nn
from typing import Sequence, Tuple

import cfg

norm_specs = [
    (0, 1746679785000, 1746680285280),
    (1, 114.04903489330026, 114.06511032518509),
    (2, 22.53559020129784, 22.552685658396516),
    (3, 296.94104248443386, 699.4823092986296)
]


class TrajPredHandler(nn.Module):
    def __init__(self, dim=cfg.dim, sample_num=cfg.pred_sample):
        super(TrajPredHandler, self).__init__()
        self.model = TrajPredLSTM(dim)
        self.norm = Norm(norm_specs[:cfg.start_dim])
        self.sample_num = sample_num

    def forward(self, x):
        x_0 = self.norm.norm(x)
        #x_tt = self.norm.unnorm(x_0)
        # mse between o_0 and x_tt
        x_1 = self.model(x_0)
        return x_0, x_1

    def gen_trajs(self, x_input, tensor_len=None, t=None):
        assert tensor_len is not None, "tensor_len must be provided during inference."
        assert t is not None, "t must be provided during inference."
        x = self.norm.norm(x_input)  # 归一化输入张量
        B, L_max, _ = x.shape
        tensor_len = torch.as_tensor(tensor_len, device=x.device, dtype=torch.long)
        # tensor_ret = torch.zeros((B, self.sample_num, L_max, x.shape[-1]), device=x.device, dtype=x.dtype)

        xx = x.unsqueeze(1).repeat(1, self.sample_num, 1, 1).view(B*self.sample_num,L_max, -1)  # 扩展维度以适应样本数
        tensor_len_ = tensor_len.unsqueeze(-1).repeat(1, self.sample_num).view(B*self.sample_num)  # 扩展 tensor_len
        tensor_ret = self.model.gen_trajs(xx, tensor_len_, t=t).view(B, self.sample_num, L_max, -1)  # 生成轨迹
        # assert a.shape == tensor_ret.shape
        # tensor_ret = a

        # for i in range(self.sample_num):
        #     tensor_len_ = tensor_len.clone()  # 每次递推都从原始 tensor_len 开始
        #     xx = x.clone()  # 每次递推都从原始 x 开始
        #     #xx = x.unsqueeze(1).repeat(1, self.sample_num, 1, 1)
        #     xx = self.model.gen_trajs(xx, tensor_len_, t=t)
        #     tensor_ret[:, i, :, :] = xx
        # 返回更新后的序列张量（含追加的 t 个向量）
        #assert (a==tensor_ret).all(), "生成的轨迹与预期不一致，请检查模型实现。"

        tensor_ret = self.norm.unnorm(tensor_ret)
        final_len = (tensor_len + t).tolist()
        traj_re_list = [tensor_ret[i, :, :final_len[i], :].contiguous() for i in range(B)]
        # aa = [a[i, :, :final_len[i], :].contiguous() for i in range(B)]
        for i in range(len(traj_re_list)):
            batch = traj_re_list[i]
            for traj in batch:
                len___ = tensor_len[i]
                assert (traj[:len___, :3] == x_input[i,:len___, :3]).all()
                assert ((traj[:len___, 3] - x_input[i,:len___, 3])**2 < 1e-8).all()
        return traj_re_list


class TrajPredLSTM(nn.Module):
    def __init__(self, dim):
        super(TrajPredLSTM, self).__init__()
        # self.baselinear = nn.Linear(2, dim)
        # self.prenorm = Norm()

        self.baseRNN = nn.LSTM(cfg.start_dim, dim, 4, batch_first=True, dropout=2e-5).to(torch.float64)
        # self.subPart = subPart(dim)
        # self.subPart2 = subPart2(dim*2)
        self.linear2startdim = nn.Linear(dim, cfg.start_dim,dtype=torch.float64)  # 用于将 LSTM 输出的维度调整为 start_dim

    def forward(self, x):
        x, a = self.baseRNN(x)
        x = self.linear2startdim(x)
        return x

    def gen_trajs(self, x, tensor_len=None, t=None):
        """
        生成轨迹，x_input 是归一化后的输入张量，tensor_len 是每个样本的长度，t 是预测步数。
        """
        assert tensor_len is not None, "tensor_len must be provided during inference."
        assert t is not None, "t must be provided during inference."
        B, L_max, _ = x.shape
        for _ in range(t):
            # 1. 前向得到预测向量
            x_1 = self(x)  # 与训练阶段保持一致
            # 2. 取出每个样本“最后一个有效位置”的预测，作为新 token
            batch_idx = torch.arange(B, device=x.device)  # (B,)
            last_valid = tensor_len - 1  # (B,)
            new_vec = x_1[batch_idx, last_valid]  # (B, D)

            # 3. 把新向量写到 x 的下一个可用槽位
            next_pos = tensor_len  # (B,)
            assert (next_pos < L_max).all(), "L_max 不够放下全部递推结果"
            x[batch_idx, next_pos,1:] = new_vec[...,1:]  # 保持时间戳不变，其他维度用新向量填充

            # 4. 更新有效长度，为下一轮递推做准备
            tensor_len = tensor_len + 1
        return x

class Norm(nn.Module):
    def __init__(self, norm_specs: Sequence[Tuple[int, float, float]] = norm_specs, ):
        super().__init__()
        # 拆分索引、min、max，并注册为 buffer，自动跟随 device
        idxs, mins, maxs = zip(*norm_specs)
        self.register_buffer('idxs', torch.tensor(idxs, dtype=torch.long))
        self.register_buffer('mins', torch.tensor(mins, dtype=torch.float64))
        self.register_buffer('ranges', torch.tensor([mx - mn for mn, mx in zip(mins, maxs)],
                                                    dtype=torch.float64))

    def norm(self, x: torch.Tensor, **kwargs):
        """
        x : (..., D) 张量，最后一维 D 至少包含需要归一化的列。
        """
        # 选出要处理的列
        cols = x[..., self.idxs]  # shape (..., k)
        cols = (cols - self.mins) / (self.ranges + 1e-12)  # 线性归一化到 [0,1]

        # 将归一化后的列写回
        x_norm = x.clone()
        x_norm[..., self.idxs] = cols

        # 交给下游函数
        test = self.unnorm(x_norm)  # 反归一化验证
        l = ((x.view(-1,cfg.start_dim) - test.view(-1,cfg.start_dim)) ** 2).mean(dim=0)
        assert (l < 1e-4).all(), "模型输出与输入不一致，请检查模型实现。"
        return x_norm

    def unnorm(self, x):
        cols = x[..., self.idxs]
        cols = cols * (self.ranges + 1e-12) + self.mins  # 反归一化
        y_denorm = x.clone()
        y_denorm[..., self.idxs] = cols
        return y_denorm
