import pickle
import torch
from torch.utils.data import DataLoader

from model.handler import ModelHandler
import cfg

def loader_collate_fn(batch):
    batch_size = len(batch)
    pre_traj, post_traj = [],[]
    for i in range(batch_size):
        pre_traj.append(batch[i][1])
        post_traj.append(batch[i][2])
    traj_s = []
    traj_s.extend(pre_traj)
    traj_s.extend(post_traj)
    d = len(traj_s[0][0]) - 1 # 去掉时间戳
    lengths = torch.tensor([len(traj) for traj in traj_s], dtype=torch.int32, device=cfg.device)
    max_len = int(lengths.max())
    padded = torch.zeros(len(traj_s), max_len, d, dtype=torch.float32, device=cfg.device)
    for i, traj in enumerate(traj_s):
        L = len(traj)
        if cfg.pad_front:
            padded[i, max_len -L:, :] = torch.tensor(traj, dtype=torch.float32, device=cfg.device)[...,1:]
        else:
            padded[i,:L,:] = torch.tensor(traj, dtype=torch.float32, device=cfg.device)[...,1:]
    return padded.reshape(2, batch_size, max_len, d), lengths.reshape(2, batch_size)

def mydiff_preproc(model_name, train_set, valid_set, db, all):
    #model = ModelHandler(cfg.dim, cfg.model_name).to(cfg.device)
    dataloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, collate_fn=loader_collate_fn)
    validloader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=loader_collate_fn)
    testloader = DataLoader(db, batch_size=cfg.batch_size, shuffle=False, collate_fn=loader_collate_fn)
    model = ModelHandler(dim=cfg.dim, model_name=model_name).to(cfg.device)
    return model, dataloader, validloader, testloader


