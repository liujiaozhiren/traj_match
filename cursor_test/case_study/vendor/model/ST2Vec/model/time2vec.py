import numpy as np
import torch
from torch import nn
import datetime

from Model import Date2VecConvert


class Date2vec_(nn.Module):
    def __init__(self):
        super(Date2vec_, self).__init__()
        self.d2v = Date2VecConvert(model_path="../model/ST2Vec/d2v_model/d2v_98291_17.169918439404636.pth")
    def forward(self, time_seq):
        all_list = []
        for one_seq in time_seq:
            one_list = []
            for timestamp in one_seq:
                t = datetime.datetime.fromtimestamp(timestamp/1000)
                t = [t.hour, t.minute, t.second, t.year, t.month, t.day]
                x = torch.Tensor(t).float()
                embed = self.d2v(x)
                one_list.append(embed)

            one_list = torch.cat(one_list, dim=0)
            one_list = one_list.view(-1, 64)

            all_list.append(one_list.numpy().tolist())

        #all_list = np.array(all_list)

        return all_list


def proc_timeseq_embedding_nodeseq(mapper, traj_list):
    d2vec = Date2vec_()
    traj_list_ts, nodeseq_list = [],[]
    for traj in traj_list:
        ts_seq = mapper.traj_to_time_seq(traj)
        node_seq = mapper.traj_to_id_seq(traj)
        traj_list_ts.append(ts_seq)
        nodeseq_list.append(node_seq)
    # timelist = mapper.traj_to_time_seq(traj_list)
    d2v = d2vec(traj_list_ts)
    return d2v, nodeseq_list