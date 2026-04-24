# pair 数据读取
import pickle
import time

import torch
from tqdm import tqdm

import cfg
from my_diffusion.fixLdiff.model import FixLenDiff
from new_test.proc import proc_multi_single_traj, clean_pairs
from new_test.proc2 import gen_con_pretrain_XY_loader

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

def do():
    train_db, db, all = getdata()
    trainloader, validloader = gen_con_pretrain_XY_loader(train_db, db)
    preDiff = FixLenDiff(lr=1e-5)  # StepwiseForwardDiffTail or FixLenDiff
    postDiff = FixLenDiff(lr=1e-5)  # StepwiseForwardDiffTail or FixLenDiff
    if torch.cuda.is_available():
        model_fn = ['/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_para_diff_pre134.68622596.pth',
                    '/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_para_diff_post134.68622596.pth']
        # model_fn = ['/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_ft_pre26.97774691_t200.pth',
        #             '/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_ft_post26.97774691_t200.pth']
        preDiff.unet.load_state_dict(torch.load(model_fn[0], map_location=cfg.device))
        postDiff.unet.load_state_dict(torch.load(model_fn[1], map_location=cfg.device))


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_train = torch.device(device)  # 训练仍在 1 号卡
    preDiff.to(device_train);
    postDiff.to(device_train)
    preDiff.train();
    postDiff.train()

    bst_dist, early_stop, model_fn = 1e7, 0, None
    for epoch in range(1, 500 + 1):
        losses_pre = losses_post = mse_pres = mse_posts = 0.0
        preDiff.train()
        postDiff.train()
        with tqdm(trainloader, desc=f"con-train epoch {epoch}") as tq:
            for trainx in tq:
                trainx_pre = trainx[:, 0]
                trainx_post = trainx[:, 1]
                trainx_pre = torch.transpose(trainx_pre, 1, 2)
                trainx_post = torch.transpose(trainx_post, 1, 2)
                loss_pre, mse_pre = preDiff.learn(trainx_pre)
                loss_post, mse_post = postDiff.learn(trainx_post)
                losses_pre += float(loss_pre)
                losses_post += float(loss_post)
                mse_pres += float(mse_pre)
                mse_posts += float(mse_post)

        # valid
        with torch.no_grad():
            dists = 0
            preDiff.eval()
            postDiff.eval()
            with tqdm(validloader, desc=f"valid epoch {epoch}") as tq:
                for trajs in tq:
                    traj_pre  = trajs[:, 0]
                    traj_post = trajs[:, 1]
                    traj_pre  = torch.transpose(traj_pre, 1, 2)
                    traj_post = torch.transpose(traj_post, 1, 2)
                    info_pre,  label_pre  = traj_pre[:, :, :cfg.diff_pre_len],  traj_pre[:, :, cfg.diff_pre_len:]
                    info_post, label_post = traj_post[:, :, :cfg.diff_pre_len], traj_post[:, :, cfg.diff_pre_len:]
                    out_pre = preDiff.infer_from_noise(info_pre)
                    out_post = postDiff.infer_from_noise(info_post)


                    d_pre = torch.linalg.norm(out_pre - label_pre, dim=1)  # (B, L)
                    d_post = torch.linalg.norm(out_post - label_post, dim=1)  # (B, L)
                    # 若作为损失：对时间和batch取均值（平均欧氏距离）
                    dist_pre = d_pre.mean()  # 标准“点间距离”的平均
                    dist_post = d_post.mean()
                    dist_item = (dist_pre + dist_post).item()
                    dists += dist_item/2
        dist = dists / len(validloader)

        if dist < bst_dist:
            bst_dist = dist
            early_stop = 0
            fn1, fn2 = f'/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_ft_pre{dist:.8f}_t{cfg.diffusion_timestamp}.pth',\
                f'/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_ft_post{dist:.8f}_t{cfg.diffusion_timestamp}.pth'
            torch.save(preDiff.unet.state_dict(), fn1)
            torch.save(postDiff.unet.state_dict(), fn2)
            print(f'saving model with valid mse {dist:.7f} at epoch {epoch}')
        else:
            early_stop += 1
            if early_stop >= 10:
                print('early stopping at epoch', epoch)
                break
        print(f'Raw PreTrain Epoch {epoch} train mse: {(mse_pres+mse_posts) / len(trainloader)}, loss: {(losses_pre +losses_post) / len(trainloader)}',
            f', Valid dist: {dist:.3f}|{bst_dist:.3f}')

        print(f"all pair{len(all)}  train/v_pair:{len(train_db)}, test pair:{len(db)} ")
if __name__ == '__main__':
    do()