import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import cfg
from match_model.my.my_infer_demo import pair_loss
from model.handler import ModelHandler
from new_test.proc import proc_multi_single_traj, clean_pairs
from new_test.sim import chk_dtw_acc, topk_acc, chk_hausdorff_acc
from pipeline.diff import HydraDataset
from raw_data_proc.search import sample_db


def getdata():
    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, 'rb'))
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    # mul_p = preproc_mul_p(mul_p)
    pairs = mul_p + sing_p
    all = clean_pairs(pairs)
    print('finish getting cleaned pairs')
    db = sample_db(all, ratio=0.1, seed=42)
    train_db = [p for p in all if p not in db]
    return train_db, db, all


def preproc_pairdb_trajs(db_pairs):
    trajs = []
    for item in db_pairs:
        mid, A, B, L_A, L_B, AA, BB = item[0], item[1], item[2], item[3], item[4], item[5], item[6]
        A, B = relocate_traj(AA, BB)
        item = [A, B]
        trajs.append(item)
    # trajs = [trajs_pre, trajs_post]
    trajs_tensor = torch.tensor(trajs, dtype=torch.float32)
    trajs_tensor = trajs_tensor.view(len(db_pairs), 2, cfg.diff_pre_len + cfg.diff_infer_len, 2).to(cfg.device)
    return trajs_tensor


def gen_traj_loader(train_db, db, shuffle=True):
    train_db_raw_traj = preproc_pairdb_trajs(train_db)
    db_raw_traj = preproc_pairdb_trajs(db)
    if cfg.device == 'cpu':
        train_db_raw_traj = train_db_raw_traj[:100]
        db_raw_traj = db_raw_traj[:100]
    dataloader = DataLoader(train_db_raw_traj, batch_size=200, shuffle=shuffle, num_workers=0)
    validloader = DataLoader(db_raw_traj, batch_size=20, shuffle=False, num_workers=0)
    return dataloader, validloader


default_shenzhen = [114.057868, 22.543099]


def relocate_traj(L_A, L_B):
    lons = [p[1] for p in L_A] + [p[1] for p in L_B]
    lats = [p[2] for p in L_A] + [p[2] for p in L_B]
    mean_lon = sum(lons) / len(lons)
    mean_lat = sum(lats) / len(lats)

    xy_L_A, xy_L_B = [], []
    for p in L_A:
        x_, y_ = p[1] - mean_lon + default_shenzhen[0], p[2] - mean_lat + default_shenzhen[1]
        x_, y_ = p[1], p[2]
        # xy_L_A.append(lonlat_to_xy_at_origin(x_, y_, default_shenzhen))
        xy_L_A.append((x_,y_))
    for p in L_B:
        x_, y_ = p[1] - mean_lon + default_shenzhen[0], p[2] - mean_lat + default_shenzhen[1]
        x_, y_ = p[1], p[2]
        # xy_L_B.append(lonlat_to_xy_at_origin(x_, y_, default_shenzhen))
        xy_L_B.append((x_,y_))
    return xy_L_A, xy_L_B


from math import cos, radians

R = 6378137.0  # WGS-84


def lonlat_to_xy_at_origin(lon: float, lat: float, origin):
    """把经纬度(lon,lat)换算成以 origin=(lon0,lat0) 为(0,0)的平面坐标(米)。x向东，y向北。"""
    lon0, lat0 = origin
    kx = R * cos(radians(lat0))  # 经度→米（在 origin 纬度处）
    dx = kx * radians(lon - lon0)
    dy = R * radians(lat - lat0)
    return dx, dy


def do():
    # data
    train_db, db, all = getdata()
    trainloader, validloader = gen_traj_loader(train_db, db, shuffle=False)
    valid_info_post = validloader.dataset[:,1]
    chk_dtw_acc(validloader)
    chk_hausdorff_acc(validloader)
    embedding = ModelHandler(dim=cfg.dim, model_name='traj2simvec').to(cfg.device)

    model_optim = torch.optim.Adam(embedding.parameters(), lr=0.0001)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_train = torch.device(device)  # 训练仍在 1 号卡

    early_stop, bst_acc = 0, 0
    for epoch in range(1, 500 + 1):
        loss_list, acc_list = [], []
        with tqdm(trainloader, f"FT epoch {epoch}") as tq:
            for trajs in tq:
                B = trajs.shape[0]
                info_pre, info_post = trajs[:, 0], trajs[:, 1]
                # hydra_pre, hydra_post = hydra_pre[:,:prefix], hydra_post[:,:prefix]
                neg_info_pre, neg_info_post = gen_neg_pair(info_pre, info_post)

                x0, x1 = info_pre, info_post

                out0 = embedding.model_pre(x0)[:, -1, :]
                out1 = embedding.model_post(x1)[:, -1, :]

                pos_dist = torch.linalg.norm(out0 - out1, dim=1)
                label_pos = torch.zeros(B, dtype=torch.long, device=info_pre.device)
                loss_pos = pair_loss(pos_dist, label_pos).mean()

                x0, x1 = neg_info_pre, neg_info_post
                out0 = embedding.model_pre(x0)[:, -1, :]
                out1 = embedding.model_post(x1)[:, -1, :]

                neg_dist = torch.linalg.norm(out0 - out1, dim=1)
                label_neg = torch.ones(B, dtype=torch.long, device=info_pre.device)
                loss_neg = pair_loss(neg_dist, label_neg).mean()

                # loss
                loss = (loss_pos + loss_neg) / 2

                model_optim.zero_grad()
                loss.backward()
                model_optim.step()

                loss_list.append(loss.item())
        with torch.no_grad():
            with tqdm(validloader, f"FT valid epoch {epoch}") as tq:
                all = torch.zeros(len(valid_info_post), len(valid_info_post)).to(cfg.device)
                num = 0
                for trajs in tq:
                    valid_info_pre = trajs[:,0]
                    x0, x1 = valid_info_pre, valid_info_post
                    x0, x1 = repeat_valid_data(x0, x1)

                    out0 = embedding.model_pre(x0)[:, -1, :]
                    out1 = embedding.model_post(x1)[:, -1, :]
                    ret = -torch.linalg.norm(out0 - out1, dim=1)

                    B = valid_info_pre.shape[0]
                    ret = ret.view(B, len(valid_info_post))
                    all[num:num + B] = ret
                    num += B
                # 计算all矩阵(shape len(v_set)*len(v_set) )每一行对角线元素是不是最大的
                acc = topk_acc(all, 1)
                top5 = topk_acc(all, 5)
                top10 = topk_acc(all, 10)
        e_loss, e_acc = np.mean(loss_list), acc
        print(f'Match Epoch {epoch}, Loss: {e_loss:.7f}, Valid Acc: {e_acc:.3f}|{bst_acc:.3f}  top5:{top5} 10:{top10}')

        if e_acc >= bst_acc:
            bst_acc = e_acc
            early_stop = 0
            fn = f'/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_ft_match{e_acc:.4f}.pth'
            # torch.save(match.state_dict(), fn)
            print(f'Best acc:{e_acc:.4f}  save model {fn}.pth')
        else:
            early_stop += 1
            if early_stop >= cfg.early_stop:
                break


def repeat_valid_data(i_pre, i_post):
    B1, B2 = i_pre.size(0), i_post.size(0)
    i_pre = i_pre.unsqueeze(1).repeat(1, B2, 1, 1).view(B1 * B2, -1, 2)
    i_post = i_post.unsqueeze(0).repeat(B1, 1, 1, 1).view(B1 * B2, -1, 2)
    return i_pre, i_post


def gen_neg_pair(info_pre, info_post):
    # 将 batch 内的样本打乱，生成负样本对
    B = info_pre.shape[0]
    idx = torch.randperm(B)
    neg_info_pre = info_pre[:]
    neg_info_post = info_post[idx]

    return neg_info_pre, neg_info_post


if __name__ == '__main__':
    do()
