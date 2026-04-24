import cfg, random

import torch
import math

default_shenzhen = [0, 0] # [114.057868, 22.543099]
def preproc(trajs, length=50):
    ret_trajs = []
    for traj in trajs:
        # traj = trajs[traj_key]
        part_trajs = clip_traj_eq_length(traj, length)
        for traj in part_trajs:
            if equal_gap_traj(traj):
                traj = relocate_traj(traj)
                ret_trajs.append(traj)
    return ret_trajs

def clip_traj_eq_length(traj, length):
    L = len(traj)
    if length <= 0:
        raise ValueError("length 必须 > 0")

    n = L // length  # 完整段数
    if n > 1:
        n=n-1
    else:
        return []

    rng = random.Random(cfg.seed)
    # 将剩余空位（slack=L-n*length）随机分配到 n+1 个空隙中（段前、段间、段后）
    slack = L - n * length
    # 选出 n 个“分隔条”的位置，范围 [1, slack+n]，排序后差分得到 n+1 个空隙
    bars = sorted(rng.sample(range(1, slack + n + 1), n))
    gaps = []
    prev = 0
    for b in bars:
        gaps.append(b - prev - 1)  # 每段前的空隙星号数
        prev = b
    gaps.append(slack + n - prev)  # 最后一个空隙

    # 计算每段的起始下标：s0=g0；si = s(i-1) + length + g_i
    starts = []
    pos = gaps[0]
    for i in range(n):
        starts.append(pos)
        if i < n - 1:
            pos += length + gaps[i + 1]

    # 切片（保持原类型）
    clips = [traj[s:s + length] for s in starts]
    return clips

def equal_gap_traj(traj):
    if len(traj) < 2:
        return False
    gap = (traj[-1][0] - traj[0][0]) // (len(traj)-1)
    for i in range(1, len(traj)):
        if abs(traj[i][0] - traj[i-1][0] - gap) > 0.33 * gap:
            return False
    return True

def relocate_traj(traj):
    lat, lon = [], []
    for p in traj:
        lon.append(p[1])
        lat.append(p[2])
    mean_lat = sum(lat) / len(lat)
    mean_lon = sum(lon) / len(lon)
    new_traj = []
    for p in traj:
        new_traj.append([p[1]-mean_lon+default_shenzhen[0], p[2]-mean_lat+default_shenzhen[1]])
    return new_traj



def mean_geo_distance(
    traj1: torch.Tensor,
    traj2: torch.Tensor,
    radius: float = 6_371_000.0,
    reduce: str = "batch",
) -> torch.Tensor:
    """
    计算两组经纬度轨迹的平均球面距离。

    参数
    ----
    traj1, traj2 : torch.Tensor
        形状 (B, 2, L)，通道 0=经度，1=纬度，单位=度。
    radius : float
        球半径，默认地球半径（米）。
    reduce : {'none','batch','all'}
        - 'none'  -> 返回逐点距离 (B, L)
        - 'batch' -> 返回每条轨迹的平均距离 (B,)
        - 'all'   -> 返回全局平均距离 (标量)

    返回
    ----
    torch.Tensor
        距离张量，取决于 reduce。
    """
    shenzhen = [114.057868, 22.543099]
    assert traj1.shape == traj2.shape and traj1.ndim == 3 and traj1.size(1) == 2, \
        f"Expect shape (B,2,L), got {traj1.shape} and {traj2.shape}"

    # 拆分经纬（度 -> 弧度）
    lon1, lat1 = traj1[:, 0, :] + shenzhen[0], traj1[:, 1, :] + shenzhen[1]
    lon2, lat2 = traj2[:, 0, :] + shenzhen[0], traj2[:, 1, :] + shenzhen[1]

    # torch.deg2rad 在较新版本可用；为了兼容，手动转换
    deg2rad = math.pi / 180.0
    lon1 = lon1 * deg2rad
    lat1 = lat1 * deg2rad
    lon2 = lon2 * deg2rad
    lat2 = lat2 * deg2rad

    # 经度差归一化到 [-π, π]，以走最短跨经线路径
    dlon = (lon2 - lon1 + math.pi) % (2 * math.pi) - math.pi
    dlat = lat2 - lat1

    # haversine 公式
    sin_dlat = torch.sin(dlat / 2.0)
    sin_dlon = torch.sin(dlon / 2.0)
    a = sin_dlat**2 + torch.cos(lat1) * torch.cos(lat2) * sin_dlon**2
    c = 2.0 * torch.atan2(torch.sqrt(a.clamp(min=0.0)), torch.sqrt((1.0 - a).clamp(min=0.0)))
    dist = radius * c  # (B, L)

    if reduce == "none":
        return dist
    elif reduce == "batch":
        return dist.mean(dim=-1)  # (B,)
    elif reduce == "all":
        return dist.mean()        # scalar
    else:
        raise ValueError("reduce must be one of {'none','batch','all'}")
