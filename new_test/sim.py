import torch, cfg
from new_test.rule import Sim
def pre_calculate_dtw(dataloader):
    sim = Sim().to(cfg.device)
    validdata = dataloader
    pre = validdata[:,0,:10]
    post = validdata[:, 1,:10].flip(dims=[1])
    N = pre.shape[0]
    dist_matrix = torch.zeros((N,N), device=cfg.device)
    for i in range(N):
        traj_pre = pre[i].unsqueeze(0).repeat(N,1,1)
        traj_post = post
        dist = sim.dtw_batch(traj_pre, traj_post)
        dist_matrix[i] = dist
    return dist_matrix

def pre_calculate_hausdorff(dataloader):
    sim = Sim().to(cfg.device)
    validdata = dataloader
    pre = validdata[:,0,:10]
    post = validdata[:, 1,:10].flip(dims=[1])
    N = pre.shape[0]
    dist_matrix = torch.zeros((N,N), device=cfg.device)
    for i in range(N):
        traj_pre = pre[i].unsqueeze(0).repeat(N,1,1)
        traj_post = post
        dist = sim.hausdorff_batch(traj_pre, traj_post)
        dist_matrix[i] = dist
    return dist_matrix

def chk_dtw_acc(dataloader):
    dist = -pre_calculate_dtw(dataloader.dataset)
    acc = topk_acc(dist, k=1)
    top5 = topk_acc(dist, k=5)
    top10 = topk_acc(dist, k=10)
    print("DTW accuracy:", acc, top5, top10)

def chk_hausdorff_acc(dataloader):
    dist = -pre_calculate_hausdorff(dataloader.dataset)
    acc = topk_acc(dist, k=1)
    top5 = topk_acc(dist, k=5)
    top10 = topk_acc(dist, k=10)
    print("Hausdorff accuracy:", acc, top5, top10)

def db_dtw_acc(db):
    trajs=[]
    for item in db:
        A, B=  item[3], item[4]
        item = [A, B]
        trajs.append(item)
    data = torch.tensor(trajs, dtype=torch.float32).to(cfg.device)

    dist = pre_calculate_dtw(data)
    acc = topk_acc(dist, k=1)
    top5 = topk_acc(dist, k=5)
    top10 = topk_acc(dist, k=10)
    print("DB DTW accuracy:", acc, top5, top10)


def topk_acc(dist, k):
    """
    dist: (B, B) 分数矩阵
    k: top-k
    """
    B = dist.shape[0]
    # 从每一行取 top-k 的 index，shape = (B, k)
    topk_idx = torch.topk(dist, k=k, dim=1).indices

    # 正确答案 index = 0,1,2,3,...,B-1
    correct = torch.arange(B, device=dist.device).unsqueeze(1)  # (B,1)

    # 判断 correct index 是否在 topk_idx 里
    hits = (topk_idx == correct).any(dim=1).float()  # (B,)

    return hits.mean().item()

def kth_smallest_no_self(mat: torch.Tensor, k: int):
    N = mat.size(0)
    vals, idx = torch.topk(-mat, k + 1, dim=1)  # idx: (N, k+1)
    row_idx = torch.arange(N, device=mat.device).unsqueeze(1)
    mask = (idx != row_idx)  # (N, k+1)
    cumsum = mask.int().cumsum(dim=1)  # (N, k+1)
    pos = (cumsum == k).int().argmax(dim=1)  # (N,)
    # 最终选择 idx[i, pos[i]]
    result = idx[torch.arange(N, device=mat.device), pos]
    return result

def get_train_sim_dict(trainloader):
    dist = pre_calculate_dtw(trainloader.dataset)
    select_pos = kth_smallest_no_self(dist, 1)
    select_neg = kth_smallest_no_self(-dist, 5)
    return select_pos, select_neg