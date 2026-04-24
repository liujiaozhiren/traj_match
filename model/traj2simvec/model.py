import torch
import torch.nn as nn
import torch.nn.functional as F

import cfg


def info_nce_loss(x: torch.Tensor,
                  temperature: float = 0.07,
                  normalize: bool = True) -> torch.Tensor:
    B, N, D = x.shape
    assert N >= 3, "Need at least 1 positive and 1 negative sample."

    # ----- split -----
    anchor    = x[:, 0]            # (B, D)
    positive  = x[:, 1]            # (B, D)
    negatives = x[:, 2:]           # (B, K, D)

    # ----- optional L2-norm -----
    if normalize:
        anchor    = F.normalize(anchor,   dim=-1)
        positive  = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives,dim=-1)

    # ----- similarity scores -----
    # dot(anchor, positive)   → (B,)
    pos_sim = torch.sum(anchor * positive, dim=-1, keepdim=True)

    # dot(anchor, negatives)  → (B, K)
    #   (B, K, D) ⋅ (B, D, 1) → (B, K, 1) → squeeze
    neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1)

    # ----- logits / temperature -----
    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature   # (B, 1+K)

    # ----- InfoNCE = cross-entropy with pos index 0 -----
    labels = torch.zeros(B, dtype=torch.long, device=x.device)    # (B,)
    loss   = F.cross_entropy(logits, labels, reduction='mean')
    return loss

class Traj2SimVec(nn.Module):
    def __init__(self, dim):
        super(Traj2SimVec, self).__init__()
        # self.baselinear = nn.Linear(2, dim)

        self.baseRNN = nn.LSTM(2, dim, 4, batch_first=True)
        self.subPart = subPart(dim)
        #self.subPart2 = subPart2(dim*2)

    def forward(self, x):
        # x = self.baselinear(x)
        # x = x[..., 1:]
        x, _ = self.baseRNN(x)
        x0 = self.subPart(x)
        #x1 = self.subPart2(x)
        return x0 #, x1


class subPart(nn.Module):
    def __init__(self, dim):
        super(subPart, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.linear3 = nn.Linear(dim, dim)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.linear2(x)
        C = nn.functional.sigmoid(x1) * nn.functional.tanh(x2)
        x3 = self.linear3(x)
        v = x + nn.functional.sigmoid(x3) * nn.functional.tanh(C)
        return v

class subPart2(nn.Module):
    def __init__(self, dim):
        super(subPart2, self).__init__()
        self.baseRNN = nn.LSTM(dim, dim, 2, batch_first=True)

    def forward(self, x):
        assert len(x.shape) == 3

        last_element = x[:, -1, :].unsqueeze(1)
        x_tmp = torch.cat([x, last_element.repeat(1, x.shape[1], 1)],dim=2)
        v, _ = self.baseRNN(x_tmp)
        return v