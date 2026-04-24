import math, copy, torch
import torch.multiprocessing as mp
from torch.utils.data import Subset, DataLoader

import cfg
from my_diffusion.fixLdiff.model import FixLenDiff
from pretrain.code.util import mean_geo_distance

VALID_WORKERS  = cfg.valid_num
VALID_GPU_LIST = list(range(VALID_WORKERS))

def _split_indices(n_items, n_chunks):
    base, rem = n_items // n_chunks, n_items % n_chunks
    out, start = [], 0
    for i in range(n_chunks):
        sz = base + (1 if i < rem else 0)
        out.append((start, start+sz))
        start += sz
    return out

def run_valid_parallel(dataset, batch_size, collate_fn, pre_state, post_state, gpu_list, valid_func=None):

    ctx = mp.get_context("spawn")
    q = ctx.SimpleQueue()

    # 复制一份权重，防止引用共享
    pre_sd  = copy.deepcopy(pre_state)
    post_sd = copy.deepcopy(post_state)

    n = len(dataset)
    ranges = _split_indices(n, len(gpu_list))

    procs = []
    for rank, (st, ed) in enumerate(ranges):
        if st == ed:
            continue
        sub_idx = list(range(st, ed))
        gpu = gpu_list[rank]
        p = ctx.Process(
            target=valid_func if valid_func is not None else _valid_worker,
            args=(sub_idx, dataset, batch_size, collate_fn, pre_sd, post_sd, gpu, q)
        )
        p.start()
        procs.append(p)

    total_loss = total_dist = 0.0
    total_cnt  = 0
    for _ in procs:
        sl, sd, c = q.get()   # 子进程只回传标量
        total_loss += sl
        total_dist += sd
        total_cnt  += c

    for p in procs:
        p.join()

    avg_loss = total_loss / max(1, total_cnt)
    avg_dist = total_dist / max(1, total_cnt)
    return avg_loss, avg_dist


import torch.nn.functional as F

def _build_models_on_device(pre_sd, post_sd, device):
    """
    TODO: 按你的项目实际构造对象（要求有 .unet 和 .infer_from_noise 接口）
    示例:
        pre = PreDiff(cfg);  post = PostDiff(cfg)
    """
    pre = FixLenDiff(device=device)   # <- 你自己的类
    post = FixLenDiff(device=device) # <- 你自己的类
    # pre.to(device)
    # post.to(device)
    # print(f'{device} in create{pre.alpha.device}')
    # print(f'{device} in create{post.alpha.device}')
    pre.unet.load_state_dict(pre_sd, strict=True)
    post.unet.load_state_dict(post_sd, strict=True)
    pre.unet.to(device).eval()
    post.unet.to(device).eval()
    # 若对象内部用到 cfg.device，可加：
    pre.device = device; post.device = device
    return pre, post

def _valid_worker(sub_idx, dataset, batch_size, collate_fn, pre_sd, post_sd, gpu, q):
    # 遵守你的约束：单线程 + spawn 环境
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False

    torch.cuda.set_device(gpu)             # 直接绑定到目标卡
    device = torch.device(f'cuda:{gpu}')

    preDiff, postDiff = _build_models_on_device(pre_sd, post_sd, device)

    sub_ds = Subset(dataset, sub_idx)
    sub_loader = DataLoader(
        sub_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True, collate_fn=collate_fn
    )

    sum_loss = sum_dist = 0.0
    cnt = 0

    with torch.inference_mode():
        for trajs in sub_loader:
            (trajs,) = trajs
            trajs = torch.transpose(trajs, 1, 2).to(device, non_blocking=True)
            reversed_trajs = torch.flip(trajs, dims=[2])

            info_pre,  label_pre  = trajs[:, :, :cfg.diff_pre_len],  trajs[:, :, cfg.diff_pre_len:]
            info_post, label_post = reversed_trajs[:, :, :cfg.diff_pre_len], reversed_trajs[:, :, cfg.diff_pre_len:]

            out_pre  = preDiff.infer_from_noise(info_pre)
            out_post = postDiff.infer_from_noise(info_post)

            loss_pre  = F.mse_loss(out_pre,  label_pre)
            loss_post = F.mse_loss(out_post, label_post)

            dist_pre  = mean_geo_distance(out_pre,  label_pre,  reduce='all')
            dist_post = mean_geo_distance(out_post, label_post, reduce='all')

            sum_loss += (loss_pre.item() + loss_post.item())
            sum_dist += (dist_pre.item() + dist_post.item())
            cnt += 1

    q.put((sum_loss, sum_dist, cnt))
