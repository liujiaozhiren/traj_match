import random
import math
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

import cfg


# ========== 数据结构回顾 ==========
# Point4   = [t, lat, lon, alt]               # 已做 (p - mid_point) 的“相对坐标”
# CleanPair = (mid, trajA, trajB)             # mid: Point4, trajA/B: List[Point4]

# ====================================================================== #
# 1) 采样 n% 的 pair 作为数据库 db
# ---------------------------------------------------------------------- #
def sample_db(pairs: List[Tuple], ratio: float, seed: int = None) -> List[Tuple]:
    """
    从 pairs 中随机采样 ratio（0~1）比例的元素。
    """
    if seed is not None:
        random.seed(seed)
    n = max(1, int(len(pairs) * ratio))
    return random.sample(pairs, n)

# ====================================================================== #
# 2) 计算 DTW —— 纯 numpy 实现，欧氏距离作 cost
# ---------------------------------------------------------------------- #
# —— 地球平均半径（米）——
R_EARTH = 6_371_000.0


def _deg2meter(dlat_deg: float, dlon_deg: float, ref_lat_deg: float) -> tuple[float, float]:
    """
    把相对差值 (Δlat, Δlon) ° 转成 (Δnorth, Δeast) 米，使用等距圆柱近似。
    ref_lat_deg：参考纬度（用 mid_point 的纬度或两轨迹平均纬度即可）
    """
    lat_rad = math.radians(ref_lat_deg)
    m_per_deg_lat = 111_132.0                              # 平均 1° 纬度≈111.132 km
    m_per_deg_lon = 111_320.0 * math.cos(lat_rad)          # 1° 经度≈111.320 km × cosφ
    return dlat_deg * m_per_deg_lat, dlon_deg * m_per_deg_lon

def dist(traj1, traj2, method, time_attr=None):
    if method == 'dtw':
        return dtw_distance(traj1, traj2, use_time=time_attr)
    elif method == 'hausdorff':
        return hausdorff_distance(traj1, traj2, use_time=time_attr)
    elif method == 'rule':
        return rule_distance(traj1, traj2, use_time=time_attr)
    else:
        raise NotImplementedError("Hausdorff distance not implemented yet.")

def to_xyz_and_t(traj, ref_lat):
    assert len(traj) > 0
    coords, ts = [], []
    if len(traj[0]) == 4:
        for dt, dlat, dlon, dalt in traj:
            if dlat>90 or dlat<-90:
                dlat, dlon = dlon, dlat
            dn, de = _deg2meter(dlat, dlon, ref_lat)
            coords.append([dn, de, dalt])
            ts.append(dt)
    elif len(traj[0]) == 3:
        for dt, dlat, dlon in traj:
            if dlat>90 or dlat<-90:
                dlat, dlon = dlon, dlat
            dn, de = _deg2meter(dlat, dlon, ref_lat)
            coords.append([dn, de, 0.0])
            ts.append(dt)
    else:
        raise ValueError("Trajectory point must be of length 3 or 4.")
    return np.asarray(coords, float), np.asarray(ts, float)


def rule_distance(traj1: List[List[float]],
                       traj2: List[List[float]],
                       ref_lat: float = None,
                       use_time = None) -> float:
    if ref_lat is None:
        ref_lat = np.mean([p[1] for p in traj1] + [p[1] for p in traj2])

    A, tA = to_xyz_and_t(traj1, ref_lat)    # A shape (m,3), tA (m,)
    B, tB = to_xyz_and_t(traj2, ref_lat)    # B shape (n,3), tB (n,)

    idxA = {t: i for i, t in enumerate(tA)}
    idxB = {t: i for i, t in enumerate(tB)}
    common_ts = sorted(set(idxA.keys()) & set(idxB.keys()))
    if not common_ts:
        # 返回最大值
        return float('inf')

    # 2. 逐点计算 3D 欧氏距离并求平均
    d_sum = 0.0
    for ts in common_ts:
        pa = A[idxA[ts]]  # (3,)
        pb = B[idxB[ts]]  # (3,)
        d_sum += np.linalg.norm(pa - pb)

    return d_sum / len(common_ts)

def hausdorff_distance(traj1: List[List[float]],
                       traj2: List[List[float]],
                       ref_lat: float = None,
                       use_time = None) -> float:
    if ref_lat is None:
        ref_lat = np.mean([p[1] for p in traj1] + [p[1] for p in traj2])


    A, tA = to_xyz_and_t(traj1, ref_lat)    # A shape (m,3), tA (m,)
    B, tB = to_xyz_and_t(traj2, ref_lat)    # B shape (n,3), tB (n,)
    m, n  = len(A), len(B)

    diff = A[:, None, :] - B[None, :, :]
    dist = np.linalg.norm(diff, axis=2)            # (m,n)

    if use_time is None:
        dt = np.abs(tA[:, None] - tB[None, :])  # (m,n)
        dist += no_time_penalty(dt)
    elif use_time['method'] == 'linear':
        dt = np.abs(tA[:, None] - tB[None, :])  # (m,n)
        dist += linear_time_penalty(dt)
    else:
        raise ValueError("use_time must be None or 'linear'.")

    hAB = dist.min(axis=1).max()               # 从 A 指向 B 的定向 Hausdorff
    hBA = dist.min(axis=0).max()               # 从 B 指向 A
    return float(max(hAB, hBA))

def linear_time_penalty(delta_t: np.ndarray, attr=0.000001) -> np.ndarray:
    return attr * delta_t

def my_time_penalty(delta_t: np.ndarray, xishu, times) -> np.ndarray:
    return (xishu * delta_t)**times

def flag_time_penalty(delta_t: np.ndarray) -> np.ndarray:
    b = delta_t > 5000
    return delta_t > 5000

def no_time_penalty(delta_t: np.ndarray) -> np.ndarray:
    return 0

def dtw_distance(
    traj1: List[List[float]],
    traj2: List[List[float]],
    ref_lat: float = None,
    use_time = None
) -> float:
    if ref_lat is None:
        ref_lat = np.mean([p[1] for p in traj1] + [p[1] for p in traj2])

    A, tA = to_xyz_and_t(traj1, ref_lat)    # A shape (m,3), tA (m,)
    B, tB = to_xyz_and_t(traj2, ref_lat)    # B shape (n,3), tB (n,)
    m, n  = len(A), len(B)

    diff = A[:, None, :] - B[None, :, :]
    dist = np.linalg.norm(diff, axis=2)            # (m,n)

    if use_time is None:
        dt = np.abs(tA[:, None] - tB[None, :])  # (m,n)
        dist += no_time_penalty(dt)
    elif use_time['method'] == 'linear':
        dt = np.abs(tA[:, None] - tB[None, :])  # (m,n)
        dist += linear_time_penalty(dt)
    elif use_time['method'] == 'my':
        dt = np.abs(tA[:, None] - tB[None, :])  # (m,n)
        dist += my_time_penalty(dt, use_time['xishu'],use_time['times'])
    else:
        raise ValueError("use_time must be None or 'linear'.")

    D = np.full((m + 1, n + 1), np.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            D[i, j] = dist[i - 1, j - 1] + min(D[i - 1, j],
                                             D[i, j - 1],
                                             D[i - 1, j - 1])
    return float(D[m, n])

# ====================================================================== #
# 3) 轨迹覆盖范围 & 密度
# ---------------------------------------------------------------------- #
def trajectory_density(db: List[Tuple]) -> float:
    """
    轨迹密度 = len(db) / 物理体积 (m³)
    体积按 db 全部点的 Δnorth、Δeast、Δalt 的范围构立方体。
    """
    if not db:
        return 0.0

    # -------------- 收集所有相对量 -----------------
    dlat, dlon, dalt, mid_lats = [], [], [], []
    for item in db:
        mid, A, B = item[0], item[3], item[4]
        mid_lats.append(mid[1])            # mid_point 的绝对纬度 (°)
        for p in A + B:
            dlat.append(p[1])              # 已经是 “Δlat(°)”
            dlon.append(p[2])
            # “Δlon(°)”
            if cfg.input_height:
                dalt.append(p[3])              # “Δalt(m)”

    # -------------- 近似 ° → m --------------------
    lat_span_deg = max(dlat) - min(dlat)
    lon_span_deg = max(dlon) - min(dlon)
    if cfg.input_height:
        alt_span_m   = max(dalt) - min(dalt)

    # 参考纬度：所有 mid_point 纬度平均
    ref_lat = 22.543099  # 默认深圳纬度
    lat_rad = math.radians(ref_lat)

    dy = lat_span_deg * 111_132.0                 # Δnorth (m)
    dx = lon_span_deg * 111_320.0 * math.cos(lat_rad)  # Δeast  (m)
    if cfg.input_height:
        dz = alt_span_m                               # Δalt    (m)

    # 若某方向跨度为 0 → ε
    eps = 1e-6
    if cfg.input_height:
        vol = max(dy, eps) * max(dx, eps) * max(dz, eps)
    square_meters = dy * dx  # 面积（m²）
    if cfg.input_height:
        return len(db) / vol, len(db) / square_meters
    return 0.0, len(db) /square_meters

# ====================================================================== #
# 4) 整合流程函数
# ---------------------------------------------------------------------- #
def analyse_pairs(db, method='dtw', time_attr=None):
    """
    • 随机抽 ratio 的 pair 做 db
    • 从 db 随机选一个 query_pair
        – 计算其 trajA 与 trajB 的 DTW
        – 再与 db 其余 pair 的 trajB 分别算 DTW
    • 计算并返回密度
    """
    # db = sample_db(pairs, ratio, seed=seed)
    # density = trajectory_density(db)

    #print(f"DB:{len(db)} 轨迹密度:{density}")
    rst = []
    with tqdm(range(len(db))) as pbar:
        for i in pbar:
            query_pair = db[i]
            mid, trajA, trajB = query_pair[0], query_pair[5], query_pair[6]
            # (1) A vs B 同对
            if is_normed(trajA):
                trajA, trajB = de_normed(trajA), de_normed(trajB)
            d_AB = dist(trajA, trajB, method, time_attr)
            # (2) A vs 其他 pair 的 B
            d_cross = []
            for p in db:
                if p is query_pair:
                    continue
                d_cross.append(dist(trajA, p[2], method, time_attr))
            min_error_d = min(d_cross, key=lambda x: x)
            if d_AB < min_error_d:
                rst.append(1)
            else:
                rst.append(0)

    acc = sum(rst)/len(rst)

    return acc

def is_normed(traj):
    if len(traj) == 0:
        return False
    if (traj[0][1] > 1.0 or traj[0][1] < -1.0) and (traj[0][2] > 1.0 or traj[0][2] < -1.0):
        return False
    return True

def de_normed(traj):
    shenzhen = [1746680000000, 114.057868, 22.543099, 500]
    denormed_traj = []
    for p in traj:
        if len(p) ==3:
            denormed_traj.append([p[0],
                                 p[1] + shenzhen[1],
                                 p[2] + shenzhen[2]])
        else:
            denormed_traj.append([p[0],
                                 p[1] + shenzhen[1],
                                 p[2] + shenzhen[2],
                                 p[3] + shenzhen[3]])

    return denormed_traj

# ====================================================================== #
# 5) 使用示例
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    pass
    # result = analyse_pairs(pairs, ratio=0.3, seed=42)
    # print("DB 内样本数:", result["db_size"])
    # print("本对 A↔B DTW:", result["dtw_AB"])
    # print("A ↔ 其它 B DTW (前5):", result["dtw_cross"][:5])
    # print("轨迹密度:", result["density"])