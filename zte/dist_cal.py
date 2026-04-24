from typing import Sequence, Any, Optional, Tuple,List
import numpy as np
import math
_EARTH_RADIUS_M = 6_371_008.8  # 平均地球半径(米)

def traj_combine(trajA, trajB, height=True):
    """
    合并两条轨迹：
      - height=True:  点为 [t, lon, lat, height, 信噪比(ps)]
      - height=False: 点为 [t, lon, lat, 信噪比]
    """
    if not isinstance(trajA, list) or not isinstance(trajB, list):
        raise TypeError("trajA/trajB 需为 list[list[...]]")

    # 字段索引
    IDX_T = 0
    IDX_LON = 1
    IDX_LAT = 2
    IDX_H = 3 if height else None
    IDX_CONF = 4 if height else 3

    def db_to_linear(db): # 这里需要和无线确认ps怎么用，这个值是 负对数 还是 正对数
        # 假设 dB 为功率标度：w = 10^(dB/10)
        if db is None or not math.isfinite(db):
            return 0.0
        return 10.0 ** (db / 10.0)

    def safe_avg(vals, ws):
        s = sum(ws)
        if s <= 0.0:
            # 全零权重时退化为简单平均
            return sum(vals) / len(vals) if vals else 0.0
        return sum(v * w for v, w in zip(vals, ws)) / s

    buckets = {}  # t -> list of points
    for p in trajA:
        buckets.setdefault(p[IDX_T], []).append(p)
    for p in trajB:
        buckets.setdefault(p[IDX_T], []).append(p)

    merged = []
    for t, points in buckets.items():
        if len(points) == 1:
            # 只有一个点，原样放入
            merged.append(list(points[0]))
            continue

        # 多个点：根据置信度（dB->线性）做加权平均
        lons = [p[IDX_LON] for p in points]
        lats = [p[IDX_LAT] for p in points]
        ws = [db_to_linear(p[IDX_CONF]) for p in points]

        lon_m = safe_avg(lons, ws)
        lat_m = safe_avg(lats, ws)

        if height:
            hs = [p[IDX_H] for p in points]
            h_m = safe_avg(hs, ws)

        w_sum = sum(ws)
        # 合并后的置信度（线性和 -> dB），避免 log10(0)
        if w_sum > 0.0:
            conf_db = 10.0 * math.log10(w_sum)
        else:
            conf_db = float("-inf")

        if height:
            merged.append([t, lon_m, lat_m, h_m, conf_db])
        else:
            merged.append([t, lon_m, lat_m, conf_db])

    # 按时间戳排序
    merged.sort(key=lambda x: x[IDX_T])
    return merged


def pre_proc(trajlist: Any, height: bool = True, timestampgap: int = 1000):
    """
    对 trajlistA 中每条轨迹的时间戳做四舍五入到 timestampgap 的整数倍；
    若同一条轨迹中因四舍五入导致时间戳重复，保留“后面的”点。
    返回与输入同结构的轨迹列表，每条轨迹按时间戳升序。

    点格式：
      - height=True  -> [t, lon, lat, h, conf]
      - height=False -> [t, lon, lat, conf]
    """
    if not isinstance(trajlist, (list, tuple)):
        raise TypeError("trajlistA 需为由多条轨迹组成的 list/tuple")
    if timestampgap <= 0:
        raise ValueError("timestampgap 必须为正整数")

    dim = 5 if height else 4
    gap = float(timestampgap)

    def round_to_gap_half_up(t: float) -> int:
        """四舍五入到 gap 的整数倍（half-up；负数也处理）。"""
        q = float(t) / gap
        r = math.floor(q + 0.5) if q >= 0 else math.ceil(q - 0.5)
        return int(r) * int(timestampgap)

    processed = []
    for ti, traj in enumerate(trajlist):
        by_ts = {}  # rounded_ts -> point(with rounded ts)
        for pi, p in enumerate(traj):
            if not isinstance(p, (list, tuple)) or len(p) != dim:
                raise ValueError(f"第 {ti} 条轨迹的第 {pi} 个点维度应为 {dim}，收到 {p}")
            t_rounded = round_to_gap_half_up(p[0])
            newp = list(p)
            newp[0] = t_rounded
            by_ts[t_rounded] = newp  # 覆盖：保留后面的点

        # 输出按时间戳升序
        merged = [by_ts[t] for t in sorted(by_ts.keys())]
        processed.append(merged)

    return processed

def traj_fusion(trajlistA, trajlistB, height=True, assign='greedy', threshold=1e5):
    trajlistA = pre_proc(trajlistA, height=height)
    trajlistB = pre_proc(trajlistB, height=height)
    traj_dist_matrix = pairwise_traj_distance_np(trajlistA, trajlistB, height=height)
    if assign == 'greedy':
        match_for_A, total_cost = greedy_assignment(traj_dist_matrix)
    else:
        raise ValueError(f"匈牙利算法需要scipy库支持，依赖GLPK库(进行改造): {assign}")
    fusion_trajs = []
    for i in range(len(match_for_A)):
        j = match_for_A[i]
        if j == -1:
            fusion_trajs.append(None)
            continue
        if traj_dist_matrix[i, j] > threshold:
            fusion_trajs.append(None)
            continue
        fusion_traj = traj_combine(trajlistA[i], trajlistB[j], height=height)
        fusion_trajs.append(fusion_traj)
    return fusion_trajs


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """球面两点大圆距离（米），输入为度。"""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlmb/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return _EARTH_RADIUS_M * c

def traj_dist(traj1: Any, traj2: Any, height: bool = True) -> float:
    """
    取共同时间戳的点做逐点距离并求平均。
    - height=True: 距离 = sqrt(地表大圆距离^2 + 高差^2)
    - height=False: 距离 = 地表大圆距离
    若无共同时间戳，返回 +inf。
    轨迹点格式：
      height=True  -> [t, lon, lat, h, conf]
      height=False -> [t, lon, lat, conf]
    """
    if not isinstance(traj1, (list, tuple)) or not isinstance(traj2, (list, tuple)):
        raise TypeError("traj1/traj2 必须是序列")

    # 建立时间戳 -> 点 的映射（题设保证同一轨迹时间戳唯一）
    map1 = {float(p[0]): p for p in traj1}
    map2 = {float(p[0]): p for p in traj2}

    common_ts = map1.keys() & map2.keys()
    if not common_ts:
        return float("inf")

    total = 0.0
    cnt = 0
    for t in common_ts:
        p1, p2 = map1[t], map2[t]
        lon1, lat1 = float(p1[1]), float(p1[2])
        lon2, lat2 = float(p2[1]), float(p2[2])
        d_surface = _haversine_m(lat1, lon1, lat2, lon2)
        if height:
            h1, h2 = float(p1[3]), float(p2[3])
            d = math.hypot(d_surface, h1 - h2)  # sqrt(d_surface^2 + dh^2)
        else:
            d = d_surface
        total += d
        cnt += 1

    return total / cnt

def pairwise_traj_distance_np(A: Sequence[Any], B: Sequence[Any],height=True) -> np.ndarray:
    m, n = len(A), len(B)
    D = np.empty((m, n), dtype=np.float64)
    for i, ta in enumerate(A):
        for j, tb in enumerate(B):
            d = traj_dist(ta, tb, height)
            D[i, j] = d
    return D

def greedy_assignment(D):
    try:
        import torch
        if isinstance(D, torch.Tensor):
            D = D.detach().cpu().numpy()
    except Exception:
        pass
    D = np.asarray(D, dtype=np.float64)
    assert D.ndim == 2, "D 必须是二维矩阵 (m x n)"
    m, n = D.shape

    rows = np.repeat(np.arange(m), n)
    cols = np.tile(np.arange(n), m)
    costs = D.reshape(-1)

    # 过滤非有限值
    valid = np.isfinite(costs)
    rows, cols, costs = rows[valid], cols[valid], costs[valid]

    # 按距离升序（稳定排序，保证可复现）
    order = np.argsort(costs, kind="stable")
    rows, cols, costs = rows[order], cols[order], costs[order]

    used_row = np.zeros(m, dtype=bool)
    used_col = np.zeros(n, dtype=bool)
    match_for_A = np.full(m, -1, dtype=int)
    total_cost = 0.0
    matched = 0
    target = min(m, n)

    for i, j, c in zip(rows, cols, costs):
        if not used_row[i] and not used_col[j]:
            used_row[i] = True
            used_col[j] = True
            match_for_A[i] = j
            total_cost += c
            matched += 1
            if matched == target:
                break

    return match_for_A, float(total_cost)

def _by_ts(traj):
    return {p[0]: p for p in traj}

if __name__ == '__main__':
    # ---------- 构造测试数据 ----------
    # A1 与 B1 在 1000/2000/3000 都对齐，坐标相同，仅高度略有差异，便于验证合并逻辑
    A1 = [
        [1000, 0.0, 0.0, 10.0, 0.0],  # conf=0 dB -> 权重=1
        [2000, 0.0, 0.0, 12.0, 0.0],  # conf=0 dB -> 权重=1
        [2999, 0.0, 0.0, 14.0, 0.0],  # 将被 pre_proc 四舍五入到 3000
    ]
    B1 = [
        [1000, 0.0, 0.0, 11.0, 0.0],  # 与 A1 同时刻：合并高度应为 (10+11)/2=10.5
        [2000, 0.0, 0.0, 22.0, 10.0],  # 这里权重=10^(10/10)=10，合并靠 B1
        [3000, 0.0, 0.0, 16.0, 0.0],  # (14+16)/2=15
    ]
    # A2 用于测试 pre_proc 的“后者覆盖前者” + 距离大使其不被匹配
    A2 = [
        [1901, 1.0, 0.0, 0.0, 0.0],  # 将四舍五入到 2000，但会被后面的 2099 覆盖
        [2099, 2.0, 0.0, 0.0, 10.0],  # -> 2000 时刻保留这条 (lon=2)
        [3000, 5.0, 0.0, 0.0, 0.0],  # 远离 B1(0°)，确保整体距离很大
    ]

    trajlistA = [A1, A2]
    trajlistB = [B1]

    # ---------- pre_proc ----------
    A_pp = pre_proc(trajlistA, height=True, timestampgap=1000)
    B_pp = pre_proc(trajlistB, height=True, timestampgap=1000)

    # A1: 2999 -> 3000
    ts_A1 = [p[0] for p in A_pp[0]]
    assert ts_A1 == [1000, 2000, 3000], f"A1 timestamps wrong: {ts_A1}"

    # A2: 1901/2099 -> 都变 2000，后者覆盖，lon 应为 2.0
    ts_A2 = [p[0] for p in A_pp[1]]
    assert ts_A2 == [2000, 3000], f"A2 timestamps wrong: {ts_A2}"
    a2_map = _by_ts(A_pp[1])
    assert math.isclose(a2_map[2000][1], 2.0), f"A2 last-wins failed: {a2_map[2000]}"

    # ---------- 距离矩阵 + 贪心匹配 ----------
    D = pairwise_traj_distance_np(A_pp, B_pp, height=True)
    assert D.shape == (2, 1), f"D shape wrong: {D.shape}"
    # 由于 A1 与 B1 很接近，而 A2 很远，应满足：
    assert D[0, 0] < D[1, 0], f"Expect D[A1,B1] < D[A2,B1], got {D}"

    match, cost = greedy_assignment(D)
    # 只有 1 列，贪心应把 B1 分给距离更小的 A1
    assert match.tolist() == [0, -1], f"Unexpected match result: {match}"

    # ---------- 融合 ----------
    fused = traj_fusion(trajlistA, trajlistB, height=True, assign='greedy', threshold=1e5)
    assert len(fused) == 2, f"fused length wrong: {len(fused)}"
    assert fused[0] is not None and fused[1] is None, f"fusion pair unexpected: {fused}"

    f0 = fused[0]
    f0_map = _by_ts(f0)
    # 合并后的时间戳应为 1000/2000/3000
    assert set(f0_map.keys()) == {1000, 2000, 3000}, f"fused ts wrong: {sorted(f0_map.keys())}"

    # 校验合并数值（权重：dB->线性；conf 合并：线性和再转 dB）
    # t=1000: 权重 1:1，高度 (10+11)/2=10.5，conf = 10*log10(1+1)=~3.0103 dB
    h_1000 = f0_map[1000][3]
    c_1000 = f0_map[1000][4]
    assert math.isclose(h_1000, 10.5, rel_tol=0, abs_tol=1e-6), f"h@1000 wrong: {h_1000}"
    assert math.isclose(c_1000, 10 * math.log10(2.0), rel_tol=0, abs_tol=1e-6), f"conf@1000 wrong: {c_1000}"

    # t=2000: 权重 1:10，高度 (12*1 + 22*10)/11 = 232/11 ≈ 21.090909
    #         conf = 10*log10(1+10) = 10*log10(11)
    h_2000 = f0_map[2000][3]
    c_2000 = f0_map[2000][4]
    assert math.isclose(h_2000, 232 / 11, rel_tol=0, abs_tol=1e-6), f"h@2000 wrong: {h_2000}"
    assert math.isclose(c_2000, 10 * math.log10(11.0), rel_tol=0, abs_tol=1e-6), f"conf@2000 wrong: {c_2000}"

    # t=3000: 权重 1:1，高度 (14+16)/2=15
    h_3000 = f0_map[3000][3]
    c_3000 = f0_map[3000][4]
    assert math.isclose(h_3000, 15.0, rel_tol=0, abs_tol=1e-6), f"h@3000 wrong: {h_3000}"
    assert math.isclose(c_3000, 10 * math.log10(2.0), rel_tol=0, abs_tol=1e-6), f"conf@3000 wrong: {c_3000}"

    # 进一步 sanity check：贪心匹配的距离确实小于阈值
    assert D[0, 0] < 1e5, f"Distance too large for fusion: {D[0, 0]}"

    print("All tests passed ✅")
    # 可选打印一些中间值
    print("D (m x n) =\n", D)
    print("match_for_A =", match.tolist(), " total_cost =", cost)
    print("fused[0] =", fused[0])
