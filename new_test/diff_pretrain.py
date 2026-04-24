import pickle
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from my_diffusion.fixLdiff.model import FixLenDiff
from new_test.parallel import _valid_worker
from new_test.util import raw_traj_clip_preproc
from pipeline.parallel import run_valid_parallel
from pipeline.subset import EpochRandomSubsetSampler
import torch.nn.functional as F

VALID_WORKERS  = 4
VALID_GPU_LIST = list(range(VALID_WORKERS))




def do(path='/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/data/chengdu/chengdu_trajs.pkl'):
    preDiff = FixLenDiff(lr=1e-5)  # StepwiseForwardDiffTail or FixLenDiff
    postDiff = FixLenDiff(lr=1e-5)  # StepwiseForwardDiffTail or FixLenDiff
    abs_path = Path(path).resolve()
    print('Resolved path =', abs_path)
    print('Exists =', abs_path.exists())
    import os
    db_path = 'tmp_db'
    if not os.path.exists(db_path):
        trajs_ = pickle.load(open(path, "rb"))
        trajs = raw_traj_clip_preproc(trajs_, length=12)
        trajs = np.array(trajs)
        traj = torch.as_tensor(trajs[:int(len(trajs) * 0.9995)]).float()
        traj_valid = torch.as_tensor(trajs[int(len(trajs) * 0.9995):]).float()
        # traj_valid = torch.as_tensor(trajs[-40:]).float()
        print(f'pretrain train size: {traj.shape}, valid size: {traj_valid.shape}')
        traj = traj[:1000000]
        tmp = traj, traj_valid

        pickle.dump(tmp, open(db_path, "wb"))
    else:
        tmp = pickle.load(open(db_path, "rb"))
        traj, traj_valid = tmp


    # # sampler
    # sampler = EpochRandomSubsetSampler(len(dataset), min(1000000, len(dataset)), 42)
    # dataloader = DataLoader(dataset, batch_size=100, sampler=sampler, num_workers=0)

    # or fix
    dataset, dataset_valid = TensorDataset(traj), TensorDataset(traj_valid)
    dataloader = DataLoader(dataset, batch_size=400, shuffle=True, num_workers=0)
    validloader = DataLoader(dataset_valid, batch_size=400, shuffle=False, num_workers=0)

    device = 'cuda:0'
    device_train = torch.device(device)  # 训练仍在 1 号卡
    preDiff.to(device_train);
    postDiff.to(device_train)
    preDiff.train();
    postDiff.train()

    _, dist = run_valid_parallel(
        dataset=validloader.dataset,
        batch_size=validloader.batch_size,
        collate_fn=getattr(validloader, 'collate_fn', None),
        pre_state=preDiff.unet.state_dict(),
        post_state=postDiff.unet.state_dict(),
        gpu_list=[1, 2, 3],
        valid_func=_valid_worker
    )

    bst_dist, early_stop, model_fn = 1e7, 0, None
    for epoch in range(1, 500 + 1):
        losses_pre = losses_post = mse_pres = mse_posts =  0.0
        preDiff.train()
        postDiff.train()
        with tqdm(dataloader, desc=f"pretrain epoch {epoch}") as tq:
            for (trainx,) in tq:
                trainx = torch.transpose(trainx, 1, 2).to(device_train, non_blocking=True)
                reversed_tranx = torch.flip(trainx, dims=[2])
                loss_pre, mse_pre = preDiff.learn(trainx)
                dist_pre, mse_post = postDiff.learn(reversed_tranx)
                losses_pre += float(loss_pre)
                mse_pres += float(mse_pre)
                mse_posts += float(mse_post)
                losses_post += float(dist_pre)

        # valid
        valid_start = time.time()

        # with torch.no_grad():
        #     dists = 0
        #     preDiff.eval()
        #     postDiff.eval()
        #     with tqdm(validloader, desc=f"valid epoch {epoch}") as tq:
        #         for (trajs,) in tq:
        #             trajs = torch.transpose(trajs, 1, 2).to(device_train)
        #             reversed_trajs = torch.flip(trajs, dims=[2])
        #             info_pre, label_pre = trajs[:, :, :10], trajs[:, :, 10:]
        #             info_post, label_post = reversed_trajs[:, :, :10], reversed_trajs[:, :,10:]
        #             out_pre = preDiff.infer_from_noise(info_pre)
        #             out_post = postDiff.infer_from_noise(info_post)
        #
        #             d_pre = torch.linalg.norm(out_pre - label_pre, dim=1)  # (B, L)
        #             d_post = torch.linalg.norm(out_post - label_post, dim=1)  # (B, L)
        #             # 若作为损失：对时间和batch取均值（平均欧氏距离）
        #             dist_pre = d_pre.mean()  # 标准“点间距离”的平均
        #             dist_post = d_post.mean()
        #             dist_item = (dist_pre + dist_post).item()
        #             dists += dist_item/2
        # dist = dists / len(validloader)

        # parallel valid

        # —— 并行验证（spawn，多卡） ——
        _, dist = run_valid_parallel(
            dataset=validloader.dataset,
            batch_size=validloader.batch_size,
            collate_fn=getattr(validloader, 'collate_fn', None),
            pre_state=preDiff.unet.state_dict(),
            post_state=postDiff.unet.state_dict(),
            gpu_list=[1,2,3],
            valid_func = _valid_worker
        )
        valid_time = (time.time() - valid_start) / 60.0
        print(f'Validation completed in {valid_time:.2f} minutes.')
        if dist < bst_dist:
            bst_dist = dist
            early_stop = 0
            fn1, fn2 = f'/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_para_diff_pre{dist:.8f}_t{cfg.diffusion_timestamp}.pth', f'/home/haitaoyuan/data/superJupterNote/traj-match/pretrain/file/model_para_diff_post{dist:.8f}_t{cfg.diffusion_timestamp}.pth'
            torch.save(preDiff.unet.state_dict(),fn1)
            torch.save(postDiff.unet.state_dict(), fn2)
            print(f'saving model with valid mse {dist:.7f} at epoch {epoch}')
        else:
            early_stop += 1
            if early_stop >= 5:
                print('early stopping at epoch', epoch)
                break
        print(f'Raw PreTrain Epoch {epoch} train mse: {(mse_pres+mse_posts) / len(dataloader)}, loss: {(losses_pre +losses_post) / len(dataloader)}',
            f', Valid dist: {dist:.3f}|{bst_dist:.3f}, time: {valid_time:.2f} mins')
    return model_fn

if __name__ == "__main__":
    import argparse, cfg
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--timestamp', type=int, default=200, help='')
    args = parser.parse_args()
    cfg.diffusion_timestamp = args.timestamp
    do()