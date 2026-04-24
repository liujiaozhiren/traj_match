import math
from typing import Sequence, Callable, Optional, List, Tuple, Dict
import cfg
# -----------------------------------------------------------
# 基本工具
# -----------------------------------------------------------
def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:

    R = 6_371_000.0                                  # 地球平均半径 (m)
    a1, a2 = math.radians(lat1), math.radians(lat2)
    da   = a2 - a1
    dl   = math.radians(lon2 - lon1)
    a = math.sin(da / 2)**2 + math.cos(a1) * math.cos(a2) * math.sin(dl / 2)**2
    return 2 * R * math.asin(math.sqrt(a))

def _point_conf(idx: int, length: int, t: int, α: float=cfg.conf_degree) -> float:
    cutoff = length - t
    return 1.0 if idx < cutoff else α ** (idx - cutoff + 1)

# -----------------------------------------------------------
# 主函数
# -----------------------------------------------------------
def overlap_mean_distance(traj1: Sequence[Sequence[float]],
                          traj2: Sequence[Sequence[float]],
                          t: int,
                          f: Optional[Callable[[float, float], float]] = None,
                          g: Optional[Callable[[float, float], float]] = None
                          ) -> Tuple[int, float]:
    if f is None:
        f = lambda c1, c2: c1 * c2
    if g is None:
        g = lambda d, c: d / c if c != 0 else float('inf')

    # -------- 把轨迹映射为 timestamp -> (idx, point) --------
    def _to_dict(traj) -> Dict[float, Tuple[int, Sequence[float]]]:
        return {int(p[0]): (i, p) for i, p in enumerate(traj)}  # 假设时间戳唯一

    dict1, dict2 = _to_dict(traj1.tolist()), _to_dict(traj2.tolist())
    common_ts = sorted(dict1.keys() & dict2.keys(), reverse=True)  # 尾部重叠，顺序无影响
    n_common = len(common_ts)
    if n_common == 0:
        return 0, 1e9  # 无重叠时间戳
        raise ValueError("轨迹没有重叠的时间戳")
        # return 0, float('nan')

    len1, len2 = len(traj1), len(traj2)
    total = 0.0

    for ts in common_ts:
        idx1, p1 = dict1[ts]
        idx2, p2 = dict2[ts]

        # --- 置信度 ---
        c1 = _point_conf(idx1, len1, t)
        c2 = _point_conf(idx2, len2, t)
        c  = f(c1, c2)

        # --- 距离 ---
        lat1, lon1, h1 = p1[1], p1[2], p1[3]
        lat2, lon2, h2 = p2[1], p2[2], p2[3]
        surface = _haversine_m(lat1, lon1, lat2, lon2)
        d_3d    = math.hypot(surface, h2 - h1)

        total += g(d_3d, c)

    return n_common, total / n_common

def sample_dist_cal(traj1_samples, traj2_samples, t):
    dist_sum, cnt = 0, 0
    for traj1 in traj1_samples:
        for traj2 in traj2_samples:
            _, dist = overlap_mean_distance(traj1, traj2, t)
            dist_sum += dist
            cnt += 1
    return dist_sum / cnt