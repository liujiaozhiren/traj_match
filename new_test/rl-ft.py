import os
# pair 数据读取
import pickle
import time

import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import cfg
from match_model.my.graph_fusion import GraphEmbedding
from match_model.my.my_infer_demo import pair_loss, batch_pairs_graph
from my_diffusion.fixLdiff.model import FixLenDiff
from new_test.match import Match
from new_test.proc import proc_multi_single_traj, clean_pairs
from new_test.proc2 import gen_con_pretrain_XY_loader
from new_test.rl_model import DeepLinUCB, get_reward_by_each, get_reward_by_simple
from new_test.sim import chk_dtw_acc, db_dtw_acc, topk_acc, get_train_sim_dict
from pipeline.diff import HydraDataset
from pipeline.load import get_hydra_graph_label

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


def get_diff_data(preDiff, postDiff, trainloader, validloader):
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
    train_dist = []
    with tqdm(trainloader, "Generating preprocessed training data") as trainloader:
        for trajs in trainloader:
            traj_pre = trajs[:, 0]
            traj_post = trajs[:, 1]
            traj_pre_ = torch.transpose(traj_pre, 1, 2)  # B C L
            traj_post_ = torch.transpose(traj_post, 1, 2)  # B C L

            info_pre, pre_label = traj_pre_[:,:, :cfg.diff_pre_len], traj_pre_[:, :, cfg.diff_pre_len:]  # B C L
            info_post, post_label = traj_post_[:, :, :cfg.diff_pre_len], traj_post_[:, :, cfg.diff_pre_len:]  # B C L

            hydra_pre = preDiff.infer_from_noise(info_pre, num=cfg.hydra_tail)  # B n C L
            hydra_post = postDiff.infer_from_noise(info_post, num=cfg.hydra_tail)  # B n C L

            pre_label = pre_label.unsqueeze(1).repeat(1, cfg.hydra_tail, 1, 1)  # B n C L
            post_label = post_label.unsqueeze(1).repeat(1, cfg.hydra_tail, 1, 1)

            d_pre = torch.linalg.norm(hydra_pre- pre_label, dim=2).mean()  # (B, L)
            d_post = torch.linalg.norm(hydra_post - post_label, dim=2).mean() # (B, L)
            dist_item = (d_pre + d_post).item()
            train_dist.append(dist_item / 2)

            train_info_post[train_num:train_num + trajs.shape[0]] = traj_post[:, :cfg.diff_pre_len]
            train_info_pre[train_num:train_num + trajs.shape[0]] = traj_pre[:, :cfg.diff_pre_len]
            train_hydra_post[train_num:train_num + trajs.shape[0]] = torch.transpose(hydra_post, 2, 3)
            train_hydra_pre[train_num:train_num + trajs.shape[0]] = torch.transpose(hydra_pre, 2, 3)
            train_num += trajs.shape[0]

    valid_num = 0
    valid_dist = []
    with tqdm(validloader, "Generating preprocessed validation data") as validloader:
        for trajs in validloader:
            traj_pre = trajs[:, 0]
            traj_post = trajs[:, 1]
            traj_pre_ = torch.transpose(traj_pre, 1, 2)  # B C L
            traj_post_ = torch.transpose(traj_post, 1, 2)  # B C L

            info_pre, pre_label = traj_pre_[:,:, :cfg.diff_pre_len], traj_pre_[:, :, cfg.diff_pre_len:]  # B C L
            info_post, post_label = traj_post_[:, :, :cfg.diff_pre_len], traj_post_[:, :, cfg.diff_pre_len:]  # B C L

            hydra_pre = preDiff.infer_from_noise(info_pre, num=cfg.hydra_tail)
            hydra_post = postDiff.infer_from_noise(info_post, num=cfg.hydra_tail)

            pre_label = pre_label.unsqueeze(1).repeat(1, cfg.hydra_tail, 1, 1)  # B n C L
            post_label = post_label.unsqueeze(1).repeat(1, cfg.hydra_tail, 1, 1)

            d_pre = torch.linalg.norm(hydra_pre- pre_label, dim=2).mean()  # (B, L)
            d_post = torch.linalg.norm(hydra_post - post_label, dim=2).mean() # (B, L)
            dist_item = (d_pre + d_post).item()
            valid_dist.append(dist_item / 2)

            valid_info_post[valid_num:valid_num + trajs.shape[0]] = traj_post[:,:cfg.diff_pre_len]
            valid_info_pre[valid_num:valid_num + trajs.shape[0]] = traj_pre[:,:cfg.diff_pre_len]
            valid_hydra_post[valid_num:valid_num + trajs.shape[0]] = torch.transpose(hydra_post, 2, 3)
            valid_hydra_pre[valid_num:valid_num + trajs.shape[0]] = torch.transpose(hydra_pre, 2, 3)
            valid_num += trajs.shape[0]

    print(f'Finish gen Diff Traj: Train Dist: {np.mean(train_dist):.7f}, Valid Dist: {np.mean(valid_dist):.7f}')

    return (train_info_pre, train_info_post, train_hydra_pre, train_hydra_post), \
        (valid_info_pre, valid_info_post, valid_hydra_pre, valid_hydra_post)


def pre_gen_ft_data(preDiff, postDiff, trainloader, validloader):
    fn = 'tmp.pkl'
    if not os.path.exists(fn):
        with torch.no_grad():
            trainset, validset = get_diff_data(preDiff, postDiff, trainloader, validloader)
            savefile = (trainset, validset)
            pickle.dump(savefile, open(fn, 'wb'))
    else:
        savefile = pickle.load(open(fn, 'rb'))
        trainset, validset = savefile

    train_info_pre, train_info_post, train_hydra_pre, train_hydra_post = trainset
    valid_info_pre, valid_info_post, valid_hydra_pre, valid_hydra_post = validset
    train_ds = HydraDataset(train_info_pre, train_info_post, train_hydra_pre, train_hydra_post)
    valid_ds = HydraDataset(valid_info_pre, valid_info_post, valid_hydra_pre, valid_hydra_post)
    valid_p = valid_info_post, valid_hydra_post[:, :cfg.rl_use]
    # 注意：如果这些张量已经在 CUDA 上，DataLoader 请用 num_workers=0
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=4, shuffle=False,
                              num_workers=0, pin_memory=False, drop_last=False)
    return train_loader, valid_loader, valid_p


def do():
    preDiff = FixLenDiff(lr=1e-5)  # StepwiseForwardDiffTail or FixLenDiff
    postDiff = FixLenDiff(lr=1e-5)  # StepwiseForwardDiffTail or FixLenDiff
    if torch.cuda.is_available():
        model_fn = [
            '/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_para_diff_pre134.68622596.pth',
            '/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_para_diff_post134.68622596.pth']
        model_fn = ['/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_ft_pre23.53854705_t200.pth',
                    '/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_ft_post23.53854705_t200.pth']
        model_fn = [f'/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_ft_pre6.46027499_t200.pth',
                    f'/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_ft_post6.46027499_t200.pth']

        preDiff.unet.load_state_dict(torch.load(model_fn[0], map_location=cfg.device))
        postDiff.unet.load_state_dict(torch.load(model_fn[1], map_location=cfg.device))
    # data
    train_db, db, all = getdata()
    trainloader, validloader = gen_con_pretrain_XY_loader(train_db, db, shuffle=False)
    chk_dtw_acc(validloader)
    # select1, select2 = get_train_sim_dict(trainloader)
    train_loader, valid_loader, valid_p = pre_gen_ft_data(preDiff, postDiff, trainloader, validloader)
    valid_info_post, valid_hydra_post = valid_p

    method = 'attn'
    prefix = 4
    match = Match(method=method, dim=cfg.match_dim, n=prefix).to(cfg.device)
    model_optim = torch.optim.Adam(match.parameters(), lr=0.00001)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if False:
        early_stop, bst_acc = 0, 0
        for epoch in range(1, 500 + 1):
            loss_list, acc_list = [], []
            with tqdm(train_loader, f"Match epoch {epoch}") as tq:
                for idx, info_pre, info_post, hydra_pre, hydra_post in tq:
                    B = info_pre.shape[0]

                    neg_info_pre, neg_info_post, neg_hydra_pre, neg_hydra_post = gen_neg_pair(
                        info_pre, info_post, hydra_pre, hydra_post)
                    label_pos = torch.ones(B, dtype=torch.long, device=info_pre.device)
                    label_neg = torch.zeros(B, dtype=torch.long, device=info_pre.device)

                    hydra_pre_base, hydra_post_base = hydra_pre[:, :prefix], hydra_post[:, :prefix]  #
                    neg_hydra_pre_base, neg_hydra_post_base = neg_hydra_pre[:, :prefix], neg_hydra_post[:, :prefix]
                    ret_base = match(info_pre, info_post, hydra_pre_base, hydra_post_base)
                    ret_neg_base = match(neg_info_pre, neg_info_post, neg_hydra_pre_base, neg_hydra_post_base)
                    loss_pos_each_base = pair_loss(ret_base, label_pos)
                    loss_neg_each_base = pair_loss(ret_neg_base, label_neg)

                    loss_pos = loss_pos_each_base.mean()
                    loss_neg = loss_neg_each_base.mean()
                    loss = (loss_pos + loss_neg) / 2

                    model_optim.zero_grad()
                    loss.backward()
                    model_optim.step()

                    loss_list.append(loss.item())
            with torch.no_grad():
                with tqdm(valid_loader, f"FT valid epoch {epoch}") as tq:
                    all = torch.zeros(len(valid_info_post), len(valid_info_post)).to(cfg.device)
                    # all2 = torch.zeros(len(valid_info_post), len(valid_info_post)).to(cfg.device)
                    num = 0
                    for _, valid_info_pre, _, valid_hydra_pre, _ in tq:
                        v_i_pre, v_i_post, v_h_pre, v_h_post = repeat_valid_data(
                            valid_info_pre, valid_info_post, valid_hydra_pre, valid_hydra_post)
                        vhpre_selbase, vhpost_selbase = v_h_pre[:, :prefix], v_h_post[:, :prefix]
                        ret_base = match(v_i_pre, v_i_post, vhpre_selbase, vhpost_selbase)
                        B1, B2 = valid_info_pre.shape[0], valid_info_post.shape[0]
                        ret_base = ret_base.view(B1, B2)
                        all[num:num + B1] = ret_base
                        num += B1
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
                torch.save(match.state_dict(), fn)
                print(f'Best acc:{e_acc:.4f}  save model {fn} ')
            else:
                early_stop += 1
                if early_stop >= cfg.early_stop:
                    break

    fn = f'/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_ft_match0.2490.pth'
    bandit = DeepLinUCB(num=prefix, d_context=2, rep_dim=32, alpha=1.0, lambda_=1.0, device=device)

    # load match model
    match.load_state_dict(torch.load(fn, map_location=cfg.device))
    match.eval()
    # freeze match model
    for param in match.parameters():
        param.requires_grad = False

    early_stop, bst_acc = 0, 0
    for epoch in range(1, 500 + 1):
        loss_list, acc_list = [], []
        with tqdm(train_loader, f"Match epoch {epoch}") as tq:
            for idx, info_pre, info_post, hydra_pre, hydra_post in tq:
                B = info_pre.shape[0]

                neg_info_pre, neg_info_post, neg_hydra_pre, neg_hydra_post = gen_neg_pair(
                    info_pre, info_post, hydra_pre, hydra_post)
                label_pos = torch.ones(B, dtype=torch.long, device=info_pre.device)
                label_neg = torch.zeros(B, dtype=torch.long, device=info_pre.device)

                match.freeze(True)
                bandit.freeze(False)
                for _ in range(6):
                    hydra_pre_base, hydra_post_base = hydra_pre[:, :prefix], hydra_post[:, :prefix] #
                    neg_hydra_pre_base, neg_hydra_post_base = neg_hydra_pre[:, :prefix], neg_hydra_post[:, :prefix]
                    ret_base = match(info_pre, info_post, hydra_pre_base, hydra_post_base)
                    ret_neg_base = match(neg_info_pre, neg_info_post, neg_hydra_pre_base, neg_hydra_post_base)
                    loss_pos_each_base = pair_loss(ret_base, label_pos)
                    loss_neg_each_base = pair_loss(ret_neg_base, label_neg)

                    loss_pos = loss_pos_each_base.mean()
                    loss_neg = loss_neg_each_base.mean()
                    loss = (loss_pos + loss_neg) / 2


                    # pos RL
                    hydra_pre_sel, hydra_post_sel = bandit.choose_step(info_pre, info_post, hydra_pre, hydra_post)
                    # reward_pos = get_reward_by_each(match, loss_pos_each_base, label_pos, info_pre, info_post, hydra_pre_base, hydra_post_base, hydra_pre_sel, hydra_post_sel)
                    reward_pos = get_reward_by_simple(match, loss_pos_each_base, label_pos, info_pre, info_post,
                                                    hydra_pre_base, hydra_post_base, hydra_pre_sel, hydra_post_sel)
                    bandit.update_batch(reward_pos)
                match.freeze(False)
                bandit.freeze(True)

                hydra_pre_pos, hydra_post_pos = bandit.choose_step(info_pre, info_post, hydra_pre, hydra_post)
                ret_pos = match(info_pre, info_post, hydra_pre_pos, hydra_post_pos)

                loss_pos_each = pair_loss(ret_pos, label_pos)
                # neg RL
                neg_hydra_pre_sel, neg_hydra_post_sel = bandit.choose_step(neg_info_pre, neg_info_post, neg_hydra_pre,
                                                                           neg_hydra_post)
                ret_neg = match(neg_info_pre, neg_info_post, neg_hydra_pre_sel, neg_hydra_post_sel)
                loss_neg_each = pair_loss(ret_neg, label_neg)
                loss_pos = loss_pos_each.mean()
                loss_neg = loss_neg_each.mean()
                loss = (loss_pos + loss_neg) / 2
                model_optim.zero_grad()
                loss.backward()
                model_optim.step()


                loss_list.append(loss.item())
        with torch.no_grad():
            with tqdm(valid_loader, f"FT valid epoch {epoch}") as tq:
                all = torch.zeros(len(valid_info_post), len(valid_info_post)).to(cfg.device)
                # all2 = torch.zeros(len(valid_info_post), len(valid_info_post)).to(cfg.device)
                num = 0
                for _, valid_info_pre, _, valid_hydra_pre, _ in tq:
                    v_i_pre, v_i_post, v_h_pre, v_h_post = repeat_valid_data(
                        valid_info_pre, valid_info_post, valid_hydra_pre, valid_hydra_post)

                    vhpre_sel, vhpost_sel = bandit.choose_step(v_i_pre, v_i_post, v_h_pre, v_h_post)
                    vhpre_selbase, vhpost_selbase = v_h_pre[:, :prefix], v_h_post[:, :prefix]
                    ret = match(v_i_pre, v_i_post, vhpre_sel, vhpost_sel)
                    #ret_base = match(v_i_pre, v_i_post, vhpre_selbase, vhpost_selbase)

                    B1, B2 = valid_info_pre.shape[0], valid_info_post.shape[0]
                    ret = ret.view(B1, B2)
                    #ret_base = ret_base.view(B1, B2)
                    all[num:num + B1] = ret
                    # all2[num:num + B1] = ret_base
                    num += B1
                # 计算all矩阵(shape len(v_set)*len(v_set) )每一行对角线元素是不是最大的
                acc = topk_acc(all, 1)
                top5 = topk_acc(all, 5)
                top10 = topk_acc(all, 10)
                # acc_base = topk_acc(all2, 1)
                # print(f'Base Valid Acc: {acc_base:.3f}')
        e_loss, e_acc = np.mean(loss_list), acc
        print(f'Match Epoch {epoch}, Loss: {e_loss:.7f}, Valid Acc: {e_acc:.3f}|{bst_acc:.3f}  top5:{top5} 10:{top10}')

        if e_acc >= bst_acc:
            bst_acc = e_acc
            early_stop = 0
            fn = f'/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_rl_{e_acc:.4f}.pth'
            torch.save(bandit.encoder.state_dict(), fn)
            print(f'Best acc:{e_acc:.4f}  save rl model {fn} ')
        else:
            early_stop += 1
            if early_stop >= cfg.early_stop:
                break
    # exit(0)
    fn = '/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_ft_match0.2421.pth'
    # load match model
    match.load_state_dict(torch.load(fn, map_location=cfg.device))

    # bandit.encoder.register(match.match)
    print(f'starting train RL only')




def repeat_valid_data(i_pre, i_post, h_pre, h_post):
    B1, B2 = i_pre.size(0) , i_post.size(0)
    i_pre = i_pre.unsqueeze(1).repeat(1, B2, 1, 1).view(B1 * B2, -1, 2)
    i_post = i_post.unsqueeze(0).repeat(B1, 1, 1, 1).view(B1 * B2, -1, 2)
    h_pre = h_pre.unsqueeze(1).repeat(1, B2, 1, 1, 1).view(B1 * B2, -1, 2, 2)
    h_post = h_post.unsqueeze(0).repeat(B1, 1, 1, 1, 1).view(B1 * B2, -1, 2, 2)
    return i_pre, i_post, h_pre, h_post

def gen_neg_pair(info_pre, info_post, hydra_pre, hydra_post):
    # 将 batch 内的样本打乱，生成负样本对
    B = info_pre.shape[0]
    idx = torch.randperm(B)
    neg_info_pre = info_pre[:]
    neg_info_post = info_post[idx]
    neg_hydra_pre = hydra_pre[:]
    neg_hydra_post = hydra_post[idx]
    label_pos = torch.ones(B, dtype=torch.long, device=info_pre.device)
    label_neg = torch.zeros(B, dtype=torch.long, device=info_pre.device)
    label = torch.cat([label_pos, label_neg], dim=0)
    return neg_info_pre, neg_info_post, neg_hydra_pre, neg_hydra_post# , label

def sim_neg_pair(idx, select, dataset):
    chose = select[idx]
    info_post = dataset.info_post[chose]
    hydra_post = dataset.hydra_post[chose]
    return info_post, hydra_post




if __name__ == '__main__':
    do()
