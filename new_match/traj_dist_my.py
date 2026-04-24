import math
from typing import List, Tuple, Dict, Optional

Point = List[float]  # 或更泛型 Any
Trajectory = List[Point]

EARTH_RADIUS_M = 6371000.0  # 平均地球半径

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # 转弧度
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return EARTH_RADIUS_M * c


def point_distance_m(
    p1: Point, p2: Point,
    lat_idx: int = 3, lon_idx: int = 4, h_idx: int = 5
) -> float:

    d_horizontal = haversine_m(p1[lat_idx], p1[lon_idx], p2[lat_idx], p2[lon_idx])
    dz = (p2[h_idx] - p1[h_idx])  # 高度(米)
    return math.hypot(d_horizontal, dz)  # sqrt(dh^2 + dz^2)


def trajectory_common_distance(
    traj1: Trajectory,
    traj2: Trajectory,
    ts_idx: int = 0, lat_idx: int = 3, lon_idx: int = 4, h_idx: int = 5,
    return_dist_list: bool = False
) -> Tuple[int, float, Optional[List[Tuple[float, float]]]]:

    # 建立时间戳索引
    d1: Dict[float, Point] = {p[ts_idx]: p for p in traj1}
    d2: Dict[float, Point] = {p[ts_idx]: p for p in traj2}

    # 时间戳交集
    common_ts = sorted(set(d1.keys()) & set(d2.keys()))
    n_common = len(common_ts)
    if n_common == 0:
        return 0, float('nan'), [] if return_dist_list else None

    dists = []
    total = 0.0
    for ts in common_ts:
        dist = point_distance_m(d1[ts], d2[ts], lat_idx=lat_idx, lon_idx=lon_idx, h_idx=h_idx)
        dists.append((ts, dist))
        total += dist

    mean_dist = total / n_common

    if return_dist_list:
        return n_common, mean_dist, dists
    else:
        return n_common, mean_dist, None
