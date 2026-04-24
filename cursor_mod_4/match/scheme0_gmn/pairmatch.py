import torch

import cfg


def _haversine(a, b, R=6371000.0):
    """
    a, b: (..., 2) -> [lon, lat] in degrees
    return: distance in meters, shape (...,)
    """
    lon1, lat1 = torch.deg2rad(a[..., 0]), torch.deg2rad(a[..., 1])
    lon2, lat2 = torch.deg2rad(b[..., 0]), torch.deg2rad(b[..., 1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    s = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    return 2 * R * torch.arcsin(torch.clamp(s.sqrt(), 0.0, 1.0))


def _pairwise_cost_reverse_mean(trajA_post, trajB_post):
    """
    trajA_post: (n, k, 2) [lon, lat]
    trajB_post: (n, k, 2) [lon, lat]
    cost[i, j] = mean_t haversine( A[i, t], B[j, k-1-t] )
    """
    assert trajA_post.ndim == 3 and trajB_post.ndim == 3
    n, k, _ = trajA_post.shape
    assert trajB_post.shape == (n, k, 2)
    # reverse B on the time axis
    B_rev = trajB_post[:, torch.arange(k - 1, -1, -1), :]  # (n, k, 2)
    # broadcast to (n, n, k, 2)
    A_exp = trajA_post[:, None, :, :]  # (n, 1, k, 2)
    B_exp = B_rev[None, :, :, :]  # (1, n, k, 2)
    dists = _haversine(A_exp, B_exp)  # (n, n, k)
    cost = dists.mean(dim=2)  # (n, n)
    return cost


def _hungarian(cost: torch.Tensor):
    """
    cost: (n, n) torch.Tensor (cpu/float)
    return: assignment list of length n, assign[i] = j
    说明：标准 O(n^3) 匈牙利；为了清晰可靠，这里用 CPU 上的实现。
    """
    c = cost.detach().cpu().numpy()
    n = c.shape[0]
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    INF = 1e30

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = c[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assign = [0] * n
    for j in range(1, n + 1):
        i = p[j] - 1
        assign[i] = j - 1
    return assign


def get_pair_map(trajA_post: torch.Tensor, trajB_post: torch.Tensor):
    """
    输入:
        trajA_post: (n, k, 2)  [lon, lat]
        trajB_post: (n, k, 2)
    输出:
        list[int] 长度 n，match[i] = j 表示 A[i] 与 B[j] 最优匹配
    """
    _ = cfg.device  # keep legacy cfg dependency behavior
    assert trajA_post.shape == trajB_post.shape
    n, _, d = trajA_post.shape
    assert d == 2, "最后一维必须是2（经纬度）"
    cost = _pairwise_cost_reverse_mean(trajA_post, trajB_post)  # (n, n)
    match = _hungarian(cost)  # list[int]
    return match

