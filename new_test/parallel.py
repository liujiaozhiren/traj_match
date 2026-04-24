import torch
from torch.utils.data import Subset, DataLoader

from my_diffusion.fixLdiff.model import FixLenDiff

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
    print('valid worker start on device', device)
    with torch.inference_mode():
        for trajs in sub_loader:
            (trajs,) = trajs
            trajs = torch.transpose(trajs, 1, 2).to(device, non_blocking=True)
            reversed_trajs = torch.flip(trajs, dims=[2])

            info_pre,  label_pre  = trajs[:, :, :10],  trajs[:, :, 10:]
            info_post, label_post = reversed_trajs[:, :, :10], reversed_trajs[:, :, 10:]

            out_pre  = preDiff.infer_from_noise(info_pre)
            out_post = postDiff.infer_from_noise(info_post)

            d_pre = torch.linalg.norm(out_pre - label_pre, dim=1)  # (B, L)
            d_post = torch.linalg.norm(out_post - label_post, dim=1)  # (B, L)
            # 若作为损失：对时间和batch取均值（平均欧氏距离）
            dist_pre = d_pre.mean()  # 标准“点间距离”的平均
            dist_post = d_post.mean()

            sum_dist += (dist_pre.item() + dist_post.item())
            cnt += 1

    q.put((sum_loss, sum_dist, cnt))