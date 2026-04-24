import torch.nn as nn
import torch


class Sim(nn.Module):
    def __init__(self, dim=128):
        super(Sim, self).__init__()
        #  self.dim = dim
        self._ = torch.nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, info_pre, info_post, hydra_pre, hydra_post):
        # info_pre, info_post: (B, L, 2)
        # hydra_pre, hydra_post: (B, n, L2, 2)
        B, n1, n2 = info_pre.size(0), hydra_pre.size(1), hydra_post.size(1)
        info_post = torch.flip(info_post, dims=[1])
        hydra_post = torch.flip(hydra_post, dims=[2])
        ret = torch.zeros((B, n1, n2), device=info_pre.device)
        for i in range(B):
            for j in range(n1):
                for k in range(n2):
                    traj_pre = torch.cat([info_pre[i], hydra_pre[i, j]], dim=0)  # L3, 2
                    traj_post = torch.cat([hydra_post[i, k], info_post[i]], dim=0)  # L3, 2
                    dist = self.dtw_single(traj_pre, traj_post)
                    ret[i, j, k] = dist
        return ret

    def fusion_sim(self, info_pre, info_post, hydra_pre, hydra_post, reduce='none'):
        B = info_pre.size(0)
        if reduce == 'min':
            ret = self.forward(info_pre, info_post, hydra_pre, hydra_post).view(B, -1)
            ret = torch.min(ret, dim=1).values
        elif reduce == 'mean':
            ret = self.forward(info_pre, info_post, hydra_pre, hydra_post).view(B, -1)
            ret = torch.mean(ret, dim=1)
        elif reduce == 'none':
            ret = self.no_tail_dtw(info_pre, info_post).view(B)
        else:
            raise ValueError(f"Unsupported reduce mode: {reduce}")
        return ret

    def no_tail_dtw(self, info_pre, info_post):
        B = info_pre.size(0)
        info_post = torch.flip(info_post, dims=[1])
        ret = self.dtw_batch(info_pre, info_post)
        # ret = torch.zeros((B,), device=info_pre.device)
        # for i in range(B):
        #     traj_pre = info_pre[i]
        #     traj_post = info_post[i]
        #     dist = self.dtw_single(traj_pre, traj_post)
        #     ret[i] = dist
        return ret

    def dtw_batch(self, trajs1, trajs2):
        B, L = trajs1.size(0), trajs1.size(1)
        cost = torch.cdist(trajs1, trajs2, p=2)
        dp = torch.empty((B, L, L), device=trajs1.device)  # (B, L, L)
        # (0,0)
        dp[:, 0, 0] = cost[:, 0, 0]
        for i in range(1, L):
            dp[:, i, 0] = cost[:, i, 0] + dp[:, i - 1, 0]
        for j in range(1, L):
            dp[:, 0, j] = cost[:, 0, j] + dp[:, 0, j - 1]
        for i in range(1, L):
            for j in range(1, L):
                m = torch.stack([ dp[:, i - 1, j], dp[:, i, j - 1], dp[:, i - 1, j - 1]], dim=1)
                m = torch.min(m, dim=1).values
                dp[:, i, j] = cost[:, i, j] + m
        return dp[:, -1, -1]

    def hausdorff_batch(self, trajs1: torch.Tensor, trajs2: torch.Tensor) -> torch.Tensor:
        cost = torch.cdist(trajs1, trajs2, p=2)  # (B, L1, L2)

        # h(A, B) = max_{a in A} min_{b in B} d(a, b)
        # 对每个 i，先在 j 上取 min，再在 i 上取 max
        min_ab = cost.min(dim=2).values  # (B, L1)
        h_ab = min_ab.max(dim=1).values  # (B,)

        # h(B, A) = max_{b in B} min_{a in A} d(a, b)
        # 对每个 j，先在 i 上取 min，再在 j 上取 max
        min_ba = cost.min(dim=1).values  # (B, L2)
        h_ba = min_ba.max(dim=1).values  # (B,)

        # Hausdorff(A, B) = max(h(A,B), h(B,A))
        H = torch.max(h_ab, h_ba)  # (B,)
        return H

    def dtw_lonlat(self, trajs1, trajs2):
        B, L = trajs1.size(0), trajs1.size(1)
        # the trajs contain lon and lat, we need cal dist
        cost = haversine_cdist_l2(trajs1, trajs2)  # (B, L, L)
        dp = torch.empty((B, L, L), device=trajs1.device)  # (B, L, L)
        # (0,0)
        dp[:, 0, 0] = cost[:, 0, 0]
        for i in range(1, L):
            dp[:, i, 0] = cost[:, i, 0] + dp[:, i - 1, 0]
        for j in range(1, L):
            dp[:, 0, j] = cost[:, 0, j] + dp[:, 0, j - 1]
        for i in range(1, L):
            for j in range(1, L):
                m = torch.stack([ dp[:, i - 1, j], dp[:, i, j - 1], dp[:, i - 1, j - 1]], dim=1)
                m = torch.min(m, dim=1).values
                dp[:, i, j] = cost[:, i, j] + m
        return dp[:, -1, -1]

    def dtw_single(self, traj1, traj2):
        L = traj1.size(0)
        assert traj2.size(0) == L
        cost = torch.cdist(traj1, traj2, p=2)
        dp = torch.empty_like(cost)  # (L, L)
        # (0,0)
        dp[0, 0] = cost[0, 0]
        for i in range(1, L):
            dp[i, 0] = cost[i, 0] + dp[i - 1, 0]
        for j in range(1, L):
            dp[0, j] = cost[0, j] + dp[0, j - 1]
        for i in range(1, L):
            for j in range(1, L):
                m = torch.stack([ dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1]]).min()
                dp[i, j] = cost[i, j] + m
        return dp[-1, -1]


def haversine_cdist_l2(a, b):
    """
    a, b: (B, L, 2) with (lon, lat) in degrees
    return: (B, L, L) great-circle distance in meters
    """
    # 1) degree -> radian
    lon1, lat1 = a[..., 0], a[..., 1]
    lon2, lat2 = b[..., 0], b[..., 1]

    lon1, lat1 = torch.deg2rad(lon1), torch.deg2rad(lat1)
    lon2, lat2 = torch.deg2rad(lon2), torch.deg2rad(lat2)

    # 2) expand for pairwise difference
    # (B, L, 1) vs (B, 1, L) -> (B, L, L)
    lon1 = lon1.unsqueeze(2)
    lat1 = lat1.unsqueeze(2)
    lon2 = lon2.unsqueeze(1)
    lat2 = lat2.unsqueeze(1)

    dlon = lon2 - lon1   # (B, L, L)
    dlat = lat2 - lat1   # (B, L, L)

    # 3) haversine formula
    R = 6371000.0  # Earth radius in meters
    h = torch.sin(dlat / 2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2)**2

    dist = 2 * R * torch.asin(torch.clamp(h.sqrt(), max=1.0))
    return dist

if __name__ == "__main__":
    sim = Sim().to('cuda')
    B, L = 61, 13
    traj1 = torch.randn(B, L, 2).to(torch.float64).to('cuda')
    traj2 = torch.randn(B, L, 2).to(torch.float64).to('cuda')
    import time
    t_start = time.time()
    ret1 = sim.dtw_batch(traj1, traj2)
    t_batch = time.time() - t_start
    t_start = time.time()
    for i in range(B):
        ret2 = sim.dtw_single(traj1[i], traj2[i])
        assert abs(ret1[i]-ret2) < 1e-4
    t_single = time.time() - t_start
    print(f"dtw_batch time: {t_batch:.4f}s, dtw_single time: {t_single:.4f}s")
        # assert abs(ret1[i]-ret2) < 1e-4
    #ret2 = sim.dtw_single(traj1[0], traj2[0])
    # print(ret1)