import argparse

# import dgl
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from pathlib import Path
import sys

from tqdm import tqdm

import cfg
import pickle
import time
import os, torch, torch.multiprocessing as mp

from match_model.my.graph_fusion import GraphEmbedding
from match_model.my.my_infer_demo import pair_loss, embedding_loss
from pipeline.diff import finetune_train2
from pipeline.load import get_hydra_graph_label
from pipeline.parallel import run_valid_parallel, VALID_GPU_LIST
from pipeline.proc_traj_pair import gen_con_pretrain_loader, preproc_mul_p, gen_ft_loader
from pipeline.subset import EpochRandomSubsetSampler
from raw_data_proc.db import clean_pairs
from raw_data_proc.my_proc import proc_multi_single_traj
from raw_data_proc.search import sample_db
# from pipeline.tqdm import tqdmu

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass
torch.set_num_threads(1)
import torch.nn.functional as F

if torch.cuda.is_available():
    # 以当前文件为基准：model.py -> fixLdiff -> my_diffusion -> traj-match(根)
    ROOT = Path('/home/haitaoyuan/data/superJupterNote/traj-match/')  # 2 就是两级父目录的上一层：traj-match
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from my_diffusion.fixLdiff.model import FixLenDiff
from pretrain.code.util import preproc, mean_geo_distance


def gen_valid(preDiff, postDiff, loader):
    preDiff.eval()
    postDiff.eval()
    sum_loss, cnt, sum_dist_ = 0.0, 0, 0.0
    sum4, sum2 = 0.0, 0.0
    # sum_dist = 0.0
    with torch.no_grad():
        with tqdm(loader, "valid_") as tq:
            for trajs in tq:
                trajs = trajs[0]
                trajs = torch.transpose(trajs, 1, 2).to(cfg.device)
                reversed_trajs = torch.flip(trajs, dims=[2])
                info_pre, label_pre = trajs[:, :, :cfg.diff_pre_len], trajs[:, :, cfg.diff_pre_len:]
                info_post, label_post = reversed_trajs[:, :, :cfg.diff_pre_len], reversed_trajs[:, :, cfg.diff_pre_len:]
                out_pre = preDiff.infer_from_noise(info_pre)
                out_post = postDiff.infer_from_noise(info_post)
                loss_pre = F.mse_loss(out_pre, label_pre)
                loss_post = F.mse_loss(out_post, label_post)
                dist_pre = mean_geo_distance(out_pre, label_pre, reduce='all')
                dist_post = mean_geo_distance(out_post, label_post, reduce='all')
                sum_loss += loss_pre.item() + loss_post.item()
                sum_dist_ += dist_pre.item() + dist_post.item()
                cnt += 1
    preDiff.train()
    postDiff.train()
    return sum_loss / cnt, sum_dist_ / cnt


def con_gen_valid(preDiff, postDiff, loader):
    preDiff.eval()
    postDiff.eval()
    sum_loss, cnt, sum_dist_ = 0.0, 0, 0.0
    with torch.no_grad():
        with tqdm(loader, "valid_") as tq:
            for trajs in tq:
                traj_pre = trajs[:, 0, :, 1:]
                traj_post = trajs[:, 1, :, 1:]
                traj_pre = torch.transpose(traj_pre, 1, 2)
                traj_post = torch.transpose(traj_post, 1, 2)
                info_pre, label_pre = traj_pre[:, :, :cfg.diff_pre_len], traj_pre[:, :, cfg.diff_pre_len:]
                info_post, label_post = traj_post[:, :, :cfg.diff_pre_len], traj_post[:, :, cfg.diff_pre_len:]
                out_pre = preDiff.infer_from_noise(info_pre)
                out_post = postDiff.infer_from_noise(info_post)
                loss_pre = F.mse_loss(out_pre, label_pre)
                loss_post = F.mse_loss(out_post, label_post)
                dist_pre = mean_geo_distance(out_pre, label_pre, reduce='all')
                dist_post = mean_geo_distance(out_post, label_post, reduce='all')
                sum_loss += loss_pre.item() + loss_post.item()
                sum_dist_ += dist_pre.item() + dist_post.item()
                cnt += 1
    preDiff.train()
    postDiff.train()
    return sum_loss / cnt, sum_dist_ / cnt


def prepare_Diff_Traj_pretrain(path=cfg.train_file):
    #  path = "../" + path
    # print('cuda available:', torch.cuda.is_available())
    preDiff = FixLenDiff()  # StepwiseForwardDiffTail or FixLenDiff
    postDiff = FixLenDiff()  # StepwiseForwardDiffTail or FixLenDiff
    abs_path = Path(path).resolve()
    print('Resolved path =', abs_path)
    print('Exists =', abs_path.exists())
    trajs = pickle.load(open(path, "rb"))

    trajs = preproc(trajs, length=12)

    # print(f'loading {len(trajs)} trajectories for training the diffusion model...')
    trajs = np.array(trajs)

    trajs[:, :, 0], trajs[:, :, 1] = trajs[:, :, 1], trajs[:, :, 0]
    traj = torch.as_tensor(trajs[:int(len(trajs) * 0.9995)]).float()
    traj_valid = torch.as_tensor(trajs[int(len(trajs) * 0.9995):]).float()

    print(f'pretrain train size: {traj.shape}, valid size: {traj_valid.shape}')
    dataset, dataset_valid = TensorDataset(traj), TensorDataset(traj_valid)
    sampler = EpochRandomSubsetSampler(len(dataset), min(1000000,len(dataset)), cfg.seed)
    dataloader = DataLoader(dataset, batch_size=cfg.pretrain_batchsize, sampler=sampler,  num_workers=0)
    validloader = DataLoader(dataset_valid, batch_size=cfg.pretrain_batchsize*4, shuffle=False, num_workers=0)

    return (preDiff, postDiff), (dataloader, validloader)
    # pretrain(trainloader=dataloader, validloader=validloader, preDiff=preDiff, postDiff=postDiff)
    # return


def pretrain(trainloader, validloader, preDiff=None, postDiff=None):
    bst_lost, bst_dist, early_stop = 1e7, 1e7, 0
    model_fn = None
    for epoch in range(1, 500 + 1):
        losses_pre, mses_pre = 0, 0
        losses_post, mses_post = 0, 0
        with tqdm(trainloader, f"pretrain epoch {epoch}") as tq:
            for _, (trainx) in enumerate(tq):
                trainx = trainx[0]
                trainx = torch.transpose(trainx, 1, 2)
                reversed_tranx = torch.flip(trainx, dims=[2])
                loss_pre, mse_pre = preDiff.learn(trainx)
                loss_post, mse_post = postDiff.learn(reversed_tranx)
                losses_pre += loss_pre
                mses_pre += mse_pre
                losses_post += loss_post
                mses_post += mse_post
        loss, dist = gen_valid(preDiff, postDiff, validloader)
        if dist < bst_dist:
            bst_dist = dist
            early_stop = 0
            torch.save(preDiff.unet.state_dict(), f'pretrain/file/model_para_diff_pre{dist:.8f}_{cfg.model_name}.pth')
            torch.save(postDiff.unet.state_dict(),f'pretrain/file/model_para_diff_post{dist:.8f}_{cfg.model_name}.pth')
            model_fn = f'pretrain/file/model_para_diff_pre{dist:.8f}_{cfg.model_name}.pth', f'pretrain/file/model_para_diff_post{dist:.8f}_{cfg.model_name}.pth'
            print(f'saving model with valid mse {dist:.7f} at epoch {epoch}')
        else:
            early_stop += 1
            if early_stop >= cfg.early_stop:
                print('early stopping at epoch', epoch)
                break
        print(
            f'Raw PreTrain Epoch {epoch}, Loss: {(losses_pre + losses_post / len(dataloader)):.7f}, Valid dist: {dist:.3f}|{bst_dist:.3f} ')
    return model_fn

def pretrain_parallel(trainloader, validloader, preDiff=None, postDiff=None, continue_train=False):
    if continue_train:
        if os.path.exists(cfg.fn_pre) and os.path.exists(cfg.fn_post):
            preDiff.unet.load_state_dict(torch.load(cfg.fn_pre), strict=True)
            postDiff.unet.load_state_dict(torch.load(cfg.fn_post), strict=True)
            print(f'Loaded existing pretrain models: {cfg.fn_pre}, {cfg.fn_post}')
            # return fn_pre, fn_post
    device_train = torch.device(cfg.device)  # 训练仍在 1 号卡
    preDiff.to(device_train); postDiff.to(device_train)
    preDiff.train(); postDiff.train()

    bst_dist, early_stop, model_fn = 1e7, 0, None

    for epoch in range(1, 500 + 1):
        losses_pre = losses_post = 0.0
        with tqdm(trainloader, desc=f"pretrain epoch {epoch}") as tq:
            for (trainx,) in tq:
                trainx = torch.transpose(trainx, 1, 2).to(device_train, non_blocking=True)
                reversed_tranx = torch.flip(trainx, dims=[2])

                loss_pre, _  = preDiff.learn(trainx)
                loss_post, _ = postDiff.learn(reversed_tranx)

                losses_pre += float(loss_pre)
                losses_post += float(loss_post)

        valid_start = time.time()
        # —— 并行验证（spawn，多卡） ——
        val_loss, val_dist = run_valid_parallel(
            dataset     = validloader.dataset,
            batch_size  = validloader.batch_size,
            collate_fn  = getattr(validloader, 'collate_fn', None),
            pre_state   = preDiff.unet.state_dict(),
            post_state  = postDiff.unet.state_dict(),
            gpu_list    = VALID_GPU_LIST
        )

        if val_dist < bst_dist:
            bst_dist = val_dist
            early_stop = 0
            fn_pre  = f'pretrain/file/model_para_diff_pre{val_dist:.8f}_{cfg.model_name}.pth'
            fn_post = f'pretrain/file/model_para_diff_post{val_dist:.8f}_{cfg.model_name}.pth'
            torch.save(preDiff.unet.state_dict(),  fn_pre)
            torch.save(postDiff.unet.state_dict(), fn_post)
            model_fn = (fn_pre, fn_post)
            print(f'[SAVE] epoch {epoch} valid dist={val_dist:.7f}')
        else:
            early_stop += 1
            if early_stop >= cfg.early_stop:
                print(f'[EARLY STOP] at epoch {epoch}')
                break
        valid_time = (time.time() - valid_start)/ 60.0
        # valid_time_min = valid_time
        print(f'Epoch {epoch}| ValidTime {valid_time}min | TrainLoss={(losses_pre+losses_post)/max(1,len(trainloader)):.6f} '
              f'| ValidDist={val_dist:.3f} | Best={bst_dist:.3f}')

    return model_fn

def con_pretrain(train_db, db, preDiff=None, postDiff=None):
    bst_lost, bst_dist, early_stop = 1e7, 1e7, 0
    model_fn = None
    trainloader, validloader = gen_con_pretrain_loader(train_db, db)
    for epoch in range(1, 500 + 1):
        losses_pre, mses_pre = 0, 0
        losses_post, mses_post = 0, 0
        with tqdm(trainloader, f"con-train epoch {epoch}") as tq:
            for trainx in tq:
                trainx_pre = trainx[:, 0, :, 1:]
                trainx_post = trainx[:, 1, :, 1:]
                trainx_pre = torch.transpose(trainx_pre, 1, 2)
                trainx_post = torch.transpose(trainx_post, 1, 2)
                loss_pre, mse_pre = preDiff.learn(trainx_pre)
                loss_post, mse_post = postDiff.learn(trainx_post)
                losses_pre += loss_pre
                mses_pre += mse_pre
                losses_post += loss_post
                mses_post += mse_post
        loss, dist = con_gen_valid(preDiff, postDiff, validloader)
        if dist < bst_dist:
            bst_dist = dist
            early_stop = 0
            torch.save(preDiff.unet.state_dict(), f'pretrain/file/model_para_diff_pre_con{dist:.8f}_{cfg.model_name}.pth')
            torch.save(postDiff.unet.state_dict(),
                       f'pretrain/file/model_para_diff_post_con{dist:.8f}_{cfg.model_name}.pth')
            print(f'saving model with valid mse {dist:.7f} at epoch {epoch}')
            model_fn = f'pretrain/file/model_para_diff_pre_con{dist:.8f}_{cfg.model_name}.pth', f'pretrain/file/model_para_diff_post_con{dist:.8f}_{cfg.model_name}.pth'
        else:
            early_stop += 1
            if early_stop >= cfg.early_stop:
                print('early stopping at epoch', epoch)
                break
        print(
            f'Continue PreTrain Epoch {epoch}, Loss: {((losses_pre + losses_post) / len(trainloader)):.7f}, Valid dist: {dist:.3f}|{bst_dist:.3f} ')

    return model_fn

def finetune_train(train_db, db, preDiff=None, postDiff=None, match=None):
    bst_acc, bst_dist, early_stop = 0, 0, 0
    loss_func = pair_loss if cfg.pair_match else embedding_loss
    trainloader, validloader, v_set = gen_ft_loader(train_db, db)


    model_optim = torch.optim.Adam(match.parameters(), lr=0.00001)
    match.load_state_dict(torch.load("model_match_0.6698762035763411_trajdiff1024.pth"), strict=True)

    for epoch in range(1, 500 + 1):
        loss_list, acc_list = [], []
        with tqdm(trainloader, f"FT epoch {epoch}") as tq:
            for trajs in tq:
                traj_pre = trajs[:, 0, :, 1:]
                traj_post = trajs[:, 1, :, 1:]
                info_pre = torch.transpose(traj_pre, 1, 2)
                info_post = torch.transpose(traj_post, 1, 2)

                hydra_pre = preDiff.infer_from_noise(info_pre, num=cfg.hydra_tail)
                hydra_post = postDiff.infer_from_noise(info_post, num=cfg.hydra_tail)

                bg_pairs, label = get_hydra_graph_label(info_pre, info_post, hydra_pre,
                                                        hydra_post, pair=cfg.pair_match)
                # TODO 这里有问题 好像不应该再flip了
                ret = match(bg_pairs, pairs=cfg.pair_match)  # [B]
                loss = loss_func(ret, label, dim=cfg.match_dim)
                model_optim.zero_grad()
                loss.backward()
                model_optim.step()
                loss_list.append(loss.item())

        with tqdm(validloader, f"FT valid epoch {epoch}") as tq:
            all = torch.zeros(len(v_set), len(v_set)).to(cfg.device)
            num = 0
            for trajs in tq:
                traj_pre = trajs[:, 0, :, 1:]
                # traj_post = trajs[:, 1, :, 1:]
                traj_post = v_set[...,1:] # N* 10 * 2
                # print("sjn ",traj_pre.shape, traj_post.shape)
                info_pre = torch.transpose(traj_pre, 1, 2)
                info_post = torch.transpose(traj_post, 1, 2)

                hydra_pre = preDiff.infer_from_noise(info_pre, num=cfg.hydra_tail)
                hydra_post = postDiff.infer_from_noise(info_post, num=cfg.hydra_tail)

                bg_pairs = get_hydra_graph_label(info_pre, info_post, hydra_pre, hydra_post,
                                      pair=cfg.pair_match, valid=True)
                ret = match(bg_pairs, pairs=cfg.pair_match)  # [B]
                B = trajs.shape[0]
                ret = ret.view(B, len(v_set))
                all[num:num + B] = ret
                num += B
            # 计算all矩阵(shape len(v_set)*len(v_set) )每一行对角线元素是不是最大的
            acc_list = []
            for i in range(len(v_set)):
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

def load_pretrain_model():
    preDiff, postDiff = FixLenDiff(),FixLenDiff()
    preDiff.unet.load_state_dict(torch.load(cfg.fn_pre), strict=True)
    postDiff.unet.load_state_dict(torch.load(cfg.fn_post), strict=True)
    print(f'Loaded existing pretrain models: {cfg.fn_pre}, {cfg.fn_post}')
    return preDiff, postDiff

if __name__ == "__main__":
    print("版本 v10.22.02.04")
    # 脚本入参 --pretrain 修改cfg.skip_pretrain = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--pretrain', action='store_true', help='Whether to perform pretraining')
    parser.add_argument('-d','--diffusion_timestamp', type=int, default=200, help='Set diffusion timestamp if provided')
    args = parser.parse_args()
    cfg.skip_pretrain = not args.pretrain
    cfg.diffusion_timestamp = args.diffusion_timestamp
    print(cfg.skip_pretrain, cfg.diffusion_timestamp)
    # 预训练
    if not cfg.skip_pretrain:
        ## 预训练数据读取
        model, data = prepare_Diff_Traj_pretrain()
        preDiff, postDiff = model
        dataloader, validloader = data
        ## 并行/串行预训练
        # model_fn = pretrain(trainloader=dataloader, validloader=validloader, preDiff=preDiff, postDiff=postDiff)
        model_fn = pretrain_parallel(trainloader=dataloader, validloader=validloader, preDiff=preDiff, postDiff=postDiff, continue_train=True)
        print('Pretrain finished, model files:', model_fn)
        preDiff.load_state_dict(torch.load(model_fn[0], map_location=cfg.device))
        postDiff.load_state_dict(torch.load(model_fn[1], map_location=cfg.device))
    else:
        preDiff, postDiff = load_pretrain_model()
    match = GraphEmbedding(node_in=2, use_edge_feat=True, rep_dim=cfg.match_dim).to(cfg.device)

    # pair 数据读取
    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, 'rb'))
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    mul_p = preproc_mul_p(mul_p)
    pairs = mul_p + sing_p
    all = clean_pairs(pairs)
    # get_dim_max_min(new_pairs)
    db = sample_db(all, ratio=0.1, seed=42)
    # density = trajectory_density(db)
    train_db = [p for p in all if p not in db]

    # print(f'only for testing, remember to del train_db[:50] and db[:50]')
    # train_db = train_db[:50]
    # db = db[:50]

    print(f"all pair{len(all)}  train/v_pair:{len(train_db)}, test pair:{len(db)} ")

    # 继续训练
    model_fn = con_pretrain(train_db, db, preDiff=preDiff, postDiff=postDiff)
    print(f'finish continue pretrain, start matching fine-tune')

    # 微调
    match.load_state_dict(torch.load("model_match_0.6698762035763411_trajdiff1024.pth"), strict=True)

    preDiff.unet.load_state_dict(torch.load("pretrain/file/model_para_diff_pre_con322.67313766_trajdiff1024.pth"), strict=True)
    postDiff.unet.load_state_dict(torch.load("pretrain/file/model_para_diff_post_con322.67313766_trajdiff1024.pth"), strict=True)
    # diff_train, diff_db = get_diff_data(preDiff, postDiff, train_db, db)
    finetune_train2(train_db, db, preDiff=preDiff, postDiff=postDiff, match=match)
