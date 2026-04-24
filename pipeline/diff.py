import os

import numpy as np
import torch
from tqdm import tqdm

import cfg
from match_model.my.my_infer_demo import pair_loss, embedding_loss
from pipeline.load import get_hydra_graph_label
from pipeline.proc_traj_pair import gen_ft_loader

from torch.utils.data import Dataset, DataLoader

class HydraDataset(Dataset):
    def __init__(self, info_pre, info_post, hydra_pre, hydra_post):
        # 形状： (N, L, 2), (N, L, 2), (N, M, L2, 2), (N, M, L2, 2)
        assert info_pre.shape[0] == info_post.shape[0] == hydra_pre.shape[0] == hydra_post.shape[0]
        self.info_pre = info_pre
        self.info_post = info_post
        self.hydra_pre = hydra_pre
        self.hydra_post = hydra_post
        self.N = info_pre.shape[0]
        self.rl_use = cfg.rl_use

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        # 返回四个张量： (L,2), (L,2), (M,L2,2), (M,L2,2)
        return (i,self.info_pre[i],
                self.info_post[i],
                self.hydra_pre[i,:self.rl_use],
                self.hydra_post[i, :self.rl_use])


def get_diff_data(preDiff, postDiff, train_db, db):
    trainloader, validloader, _ = gen_ft_loader(train_db, db, shuffle=False)
    train_size, valid_size = len(trainloader.dataset), len(validloader.dataset)
    train_info_pre = torch.zeros((train_size, cfg.diff_pre_len, 2), device=cfg.device)
    train_info_post = torch.zeros((train_size, cfg.diff_pre_len, 2), device=cfg.device)
    valid_info_pre = torch.zeros((valid_size, cfg.diff_pre_len, 2), device=cfg.device)
    valid_info_post = torch.zeros((valid_size, cfg.diff_pre_len, 2), device=cfg.device)

    train_hydra_pre = torch.zeros((train_size, cfg.hydra_tail, cfg.diff_infer_len, 2), device=cfg.device)
    train_hydra_post = torch.zeros((train_size, cfg.hydra_tail, cfg.diff_infer_len, 2), device=cfg.device)
    valid_hydra_pre = torch.zeros((valid_size, cfg.hydra_tail, cfg.diff_infer_len, 2), device=cfg.device)
    valid_hydra_post = torch.zeros((valid_size, cfg.hydra_tail, cfg.diff_infer_len, 2), device=cfg.device)

    train_num = 0
    with tqdm(trainloader, "Generating preprocessed training data") as trainloader:
        for trajs in trainloader:
            traj_pre = trajs[:, 0, :, 1:]
            traj_post = trajs[:, 1, :, 1:]
            info_pre = torch.transpose(traj_pre, 1, 2) # B C L
            info_post = torch.transpose(traj_post, 1, 2) # B C L
            hydra_pre = preDiff.infer_from_noise(info_pre, num=cfg.hydra_tail) # B n C L
            hydra_post = postDiff.infer_from_noise(info_post, num=cfg.hydra_tail) # B n C L

            train_info_post[train_num:train_num+trajs.shape[0]] = traj_post
            train_info_pre[train_num:train_num+trajs.shape[0]] = traj_pre
            train_hydra_post[train_num:train_num+trajs.shape[0]] = torch.transpose(hydra_post,2,3)
            train_hydra_pre[train_num:train_num+trajs.shape[0]] = torch.transpose(hydra_pre,2,3)
            train_num += trajs.shape[0]

    train_num = 0
    with tqdm(validloader, "Generating preprocessed validation data") as validloader:
        for trajs in validloader:
            traj_pre = trajs[:, 0, :, 1:]
            traj_post = trajs[:, 1, :, 1:]
            info_pre = torch.transpose(traj_pre, 1, 2)
            info_post = torch.transpose(traj_post, 1, 2)
            hydra_pre = preDiff.infer_from_noise(info_pre, num=cfg.hydra_tail)
            hydra_post = postDiff.infer_from_noise(info_post, num=cfg.hydra_tail)
            valid_info_post[train_num:train_num+trajs.shape[0]] = traj_post
            valid_info_pre[train_num:train_num+trajs.shape[0]] = traj_pre
            valid_hydra_post[train_num:train_num+trajs.shape[0]] = torch.transpose(hydra_post,2,3)
            valid_hydra_pre[train_num:train_num+trajs.shape[0]] = torch.transpose(hydra_pre,2,3)
            train_num += trajs.shape[0]

    return (train_info_pre, train_info_post, train_hydra_pre, train_hydra_post), \
              (valid_info_pre, valid_info_post, valid_hydra_pre, valid_hydra_post)

def gen_preproced_loader(preDiff, postDiff, train_db, db):
    tmp_fn = f'preproced_ft_data_{len(db)}.pt'
    if os.path.exists(tmp_fn):
        print(f'Loading preprocessed data from {tmp_fn} ...')
        data = torch.load(tmp_fn)
        trainset, validset = data['trainset'], data['validset']
        print('Data loaded.')
    else:
        trainset, validset = get_diff_data(preDiff, postDiff, train_db, db)
        torch.save({'trainset': trainset, 'validset': validset}, tmp_fn)

    train_info_pre, train_info_post, train_hydra_pre, train_hydra_post = trainset
    valid_info_pre, valid_info_post, valid_hydra_pre, valid_hydra_post = validset
    train_ds = HydraDataset(train_info_pre, train_info_post, train_hydra_pre, train_hydra_post)
    valid_ds = HydraDataset(valid_info_pre, valid_info_post, valid_hydra_pre, valid_hydra_post)
    valid_p =  valid_info_post, valid_hydra_post
    # 注意：如果这些张量已经在 CUDA 上，DataLoader 请用 num_workers=0
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=10, shuffle=False,
                              num_workers=0, pin_memory=False, drop_last=False)
    return train_loader, valid_loader, valid_p


def finetune_train2(train_db, db, preDiff=None, postDiff=None, match=None):
    bst_acc, bst_dist, early_stop = 0, 0, 0
    loss_func = pair_loss if cfg.pair_match else embedding_loss
    train_loader, valid_loader, valid_p = gen_preproced_loader(preDiff, postDiff, train_db, db)
    valid_info_post, valid_hydra_post = valid_p
    print(f'valid_info_post:{valid_info_post.shape}, valid_hydra_post:{valid_hydra_post.shape}')

    model_optim = torch.optim.Adam(match.parameters(), lr=0.00001)


    for epoch in range(1, 500 + 1):
        loss_list, acc_list = [], []
        with tqdm(train_loader, f"FT epoch {epoch}") as tq:
            for info_pre, info_post, hydra_pre, hydra_post in tq:
                bg_pairs, label = get_hydra_graph_label(info_pre, info_post, hydra_pre, hydra_post, pair=cfg.pair_match)

                ret = match(bg_pairs, pairs=cfg.pair_match)  # [B]
                loss = loss_func(ret, label, dim=cfg.match_dim)
                model_optim.zero_grad()
                loss.backward()
                model_optim.step()
                loss_list.append(loss.item())
        with torch.no_grad():
            with tqdm(valid_loader, f"FT valid epoch {epoch}") as tq:
                all = torch.zeros(len(valid_info_post), len(valid_info_post)).to(cfg.device)
                num = 0
                for valid_info_pre, _, valid_hydra_pre, _ in tq:
                    bg_pairs = get_hydra_graph_label(valid_info_pre, valid_info_post, valid_hydra_pre, valid_hydra_post, pair=cfg.pair_match, valid=True)
                    ret = match(bg_pairs, pairs=cfg.pair_match)  # [B]
                    B = valid_info_pre.shape[0]
                    ret = ret.view(B, len(valid_info_post))
                    all[num:num + B] = ret
                    num += B
                # 计算all矩阵(shape len(v_set)*len(v_set) )每一行对角线元素是不是最大的
                acc_list = []
                for i in range(len(valid_info_post)):
                    row = all[i]
                    # 计算 row[i] 是不是最大的
                    if row[i] == torch.max(row):
                        acc_list.append(1)
                    else:
                        acc_list.append(0)
        e_loss, e_acc = np.mean(loss_list), np.mean(acc_list)
        print(f'Finetune Epoch {epoch}, Loss: {e_loss:.7f}, Valid Acc: {e_acc:.3f}|{bst_acc:.3f} ')

        if e_acc >= bst_acc:
            bst_acc = e_acc
            early_stop = 0
            torch.save(match.state_dict(), f'model_match_{e_acc}_{cfg.model_name}.pth')
            bst_pth = f'model_match_{e_acc}_{cfg.model_name}.pth'
            print(f'Best acc:{e_acc:.4f}  save model_para_{e_acc}_{cfg.model_name}.pth')
        else:
            early_stop += 1
            if early_stop >= cfg.early_stop:
                break
            # print(f'worse acc:{acc:.4f}')
        # print(f'HR5:{bst5:.4f},HR10:{bst10:.4f}, HR50:{bst50:.4f}, NDCG:{bndcg:.7f}, HR10in50:{bst10in50:.4f}')

    # model.load_state_dict(torch.load(bst_pth, map_location=cfg.device))

    return