import random
import numpy as np
from typing import List, Tuple
import copy
import statistics


def multi_traj_match_fetch(key, item):
    min_t = float('inf')
    timelist_all = []
    t_all_new = []
    trajs = item

    for i in range(len(trajs)):
        traj = trajs[i]
        new_traj = []
        for p in traj:
            new_traj.append(p[:8])  # 只保留前8个数值
        trajs[i] = new_traj
    traj1 = trajs[0]
    traj2 = trajs[1]
    if key == '00460000-00c2-62ab-196a-e6a1ef400000':
        mid_p = cmb_point(traj1[:2], traj2[-2:])
        return [(mid_p, traj1, traj2)]
    elif key == '00460000-00c2-62ab-196a-e6c2e4c00000':
        mid1 = cmb_point(traj1[48:50], traj2)
        p1 = (mid1, traj1[:51], traj2)
        mid2 = cmb_point(traj1[50:52], traj2)
        p2 = (mid2, traj1[49:], traj2)
        return [p1, p2]
    elif key == '00460000-00c2-6298-196a-e84231c00000':
        traj3 = trajs[2]
        mid1 = cmb_point(traj1[10:12], traj2[:2])
        p1 = (mid1, traj1[:12], traj2)
        mid2 = cmb_point(traj2[-3:], traj3[:3])
        p2 = (mid2, traj2, traj3)
        mid3 = cmb_point(traj3[-2:], traj1[12:14])
        p3 = (mid3, traj3, traj1[12:])
        return [p1, p2, p3]
    elif key == '00460000-00c2-62ab-196a-e874bb400001':
        mid1 = cmb_point(traj1[13:15], traj2[:2])
        p1 = (mid1, traj1[:15], traj2)
        mid2 = cmb_point(traj1[15:17], traj2[-2:])
        p2 = (mid2, traj1[15:], traj2)
        return [p1, p2]
    elif key == '00460000-00c2-6298-196a-e8be76400000':
        mid1 = cmb_point(traj1[11:13], traj2[:2])
        p1 = (mid1, traj1[:13], traj2)
        mid2 = cmb_point(traj1[13:15], traj2[-2:])
        p2 = (mid2, traj1[13:], traj2)
        return [p1, p2]
    elif key == '00460000-00c2-62ab-196a-e61222c00000':
        mid1 = cmb_point(traj1[24:26], traj2[:2])
        p1 = (mid1, traj1[:26], traj2[:-1])
        mid2 = cmb_point(traj1[26:28], traj2[-3:-1])
        p2 = (mid2, traj1[26:35], traj2[:-1])
        mid3 = cmb_point(traj1[33:35], traj2[-1:])
        p3 = (mid3, traj1[26:35], traj2[-1:])
        mid4 = cmb_point(traj1[35:37], traj2[-1:])
        p4 = (mid4, traj2[-1:], traj1[35:])
        return [p1, p2, p3, p4]
    elif key == '00460000-00c2-62ab-196a-e6eb2d400000':
        mid1 = cmb_point(traj1[53:55], traj2[:2])
        p1 = (mid1, traj1[:55], traj2)
        mid2 = cmb_point(traj1[55:57], traj2[-2:])
        p2 = (mid2, traj1[55:], traj2)
        return [p1, p2]
    elif key == '00460000-00c2-62ab-196a-e71181c00000':
        # 2      00-05       06-11       12-20
        #      411 417    421 426      431 439
        # 1 00-48   49-51        52-55       56-64
        mid1 = cmb_point(traj1[47:49], traj2[0:2])
        p1 = (mid1, traj1[0:49], traj2[0:6])

        mid2 = cmb_point(traj1[49:51], traj2[4:6])
        p2 = (mid2, traj1[49:52], traj2[0:6])

        mid3 = cmb_point(traj1[50:52], traj2[6:8])
        p3 = (mid3, traj1[49:52], traj2[6:12])

        mid4 = cmb_point(traj1[52:54], traj2[10:12])
        p4 = (mid4, traj1[52:56], traj2[6:12])

        mid5 = cmb_point(traj1[54:56], traj2[12:14])
        p5 = (mid5, traj1[52:56], traj2[12:])

        mid6 = cmb_point(traj1[56:58], traj2[19:21])
        p6 = (mid6, traj1[56:], traj2[12:])
        return [p1, p2, p3, p4, p5, p6]
    elif key == '00460000-00c2-62ab-196a-e7bc29400000':
        mid1 = cmb_point(traj1[63:69], traj2[:])
        p1 = (mid1, traj1[:69], traj2)
        mid2 = cmb_point(traj1[69:71], traj2[-2:])
        p2 = (mid2, traj1[69:], traj2)
        return [p1, p2]
    elif key == '00460000-00c2-62ab-196a-e7ea4dc00000':
        mid1 = cmb_point(traj1[:-2], traj2[:2])
        p1 = (mid1, traj1, traj2)
        return [p1]
    elif key == '00460001-0040-8794-9680-88751c297a62':
        mid1 = cmb_point(traj1[:-2], traj2[:2])
        p1 = (mid1, traj1, traj2)
        return [p1]
    elif key == '00460001-0040-8794-9680-88e27c28aa6b':  ### TODO 8条轨迹拆分
        # 0 00-19                20-51               52-
        #    131876         131936 132286       132326
        # 1           0-4                                            5-
        #       131886 131926                               132986
        # 2                              0-6
        #                          132216 132316

        traj3 = trajs[2]
        mid1 = cmb_point(traj1[18:20], traj2[:2])
        p1 = (mid1, traj1[0:20], traj2[:5])
        mid2 = cmb_point(traj1[20:22], traj2[3:5])
        p2 = (mid2, traj1[20:52], traj2[:5])
        mid3 = cmb_point(traj1[45:52], traj3[:7])
        p3 = (mid3, traj1[20:52], traj3[:7])
        mid4 = cmb_point(traj3[:7], traj1[52:55])
        p4 = (mid4, traj3[:7], traj1[52:52 + 15])

        return [p1, p2, p3, p4]
    elif key == '00460001-0040-8796-9680-8a8d0c2cfc07':  ### TODO 2条轨迹拆分
        # 0 00-14                15-17_19-54                     55-63
        #    3290               3308  3350                    3417 3426
        # 1           0-18          19       20-45
        #         3286 3310      3325    3351 3376                             3438
        # 2                                           0-39                 40-43
        #                                           3377 3416           3423-3429
        traj3 = trajs[2]
        mid1 = cmb_point(traj1[12:15], traj2[:3])
        p1 = (mid1, traj1[0:15], traj2[:19])

        new_traj1 = traj1[15:18] + traj1[19:55]

        mid2 = cmb_point(traj1[15:18], traj2[16:19])
        p2 = (mid2, new_traj1, traj2[:19])

        mid3 = cmb_point(traj1[52:55], traj2[20:23])
        p3 = (mid3, new_traj1, traj2[20:46])

        mid4 = cmb_point(traj3[:3], traj2[43:46])
        p4 = (mid4, traj3[:40], traj2[20:46])

        mid5 = cmb_point(traj3[37:40], traj1[55:58])
        p5 = (mid5, traj3[:40], traj1[55:64])

        mid6 = cmb_point(traj3[40:44], traj1[60:64])
        p6 = (mid6, traj3[40:44], traj1[55:64])

        return [p1, p2, p3, p4, p5, p6]
    elif key == '00460001-0040-8796-9680-8d64642d0c10':
        mid1 = cmb_point(traj1[:-2], traj2[1:3])
        p1 = (mid1, traj1, traj2[1:])
        return [p1]
    elif key == '00460000-00c2-3d81-196d-2cf8cac00000':
        mid1 = cmb_point(traj1[:-6], traj2[4:])
        p1 = (mid1, traj1, traj2)
        return [p1]
    elif key == '00460000-00c2-3d81-196d-2d2578400000':
        mid1 = cmb_point(traj1[:-4], traj2[4:])
        p1 = (mid1, traj1, traj2)
        return [p1]
    elif key == '00460000-00c3-048e-9646-de81bc1c2f77':
        mid1 = cmb_point(traj1[:-4], traj2[4:])
        p1 = (mid1, traj1, traj2)
        return [p1]
    elif key == '00460000-00c3-048e-9646-e05e4c1c9fad':
        mid1 = cmb_point(traj1[:-2], traj2)
        p1 = (mid1, traj1, traj2)
        return [p1]
    elif key == '00460000-00c3-048e-9646-e0855c1dafb3':
        mid1 = cmb_point(traj1[10:14], traj2[:4])
        p1 = (mid1, traj1[:14], traj2)
        mid2 = cmb_point(traj1[14:18], traj2[-4:])
        p2 = (mid2, traj1[14:], traj2)
        return [p1, p2]
    elif key == '00460000-00c2-3d81-1964-6e26da400000':
        return []
    else:
        mid_p = []
    return mid_p


def cmb_point(traj1, traj2):
    """
    Combine two trajectories into a single trajectory by averaging their points.
    Assumes both trajectories are sorted by time.
    """
    combined1 = [0, 0, 0, 0, 0, 0, 0, 0]
    i, j = 0, 0
    for p1 in traj1:
        for i, v in enumerate(p1[:8]):
            combined1[i] += v
    for i in range(len(combined1)):
        combined1[i] /= len(traj1)
    combined2 = [0, 0, 0, 0, 0, 0, 0, 0]
    for p2 in traj2:
        for i, v in enumerate(p2[:8]):
            combined2[i] += v
    for i in range(len(combined2)):
        combined2[i] /= len(traj2)
    combined = [0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(combined2)):
        combined[i] = (combined1[i] + combined2[i]) / 2
    return combined


###############################################################################
# single split
###############################################################################

def _trunc_norm_int(mu: float, sigma: float, low: int, high: int) -> int:
    """截断正态取整。"""
    while True:
        v = int(random.gauss(mu, sigma))
        if low <= v <= high:
            return v


def _sample_core_lengths(total_len: int,
                         mu: int = 20,
                         low: int = 6,
                         high: int = 35) -> List[int]:
    """
    按截断正态分布拆分“核心段”长度（不含重叠区）。
    取 high=35，是为了后续 +5 仍不超 40。
    """
    core_lens, remain = [], total_len
    while remain > 0:
        if remain < low and core_lens:
            core_lens[-1] += remain
            break
        ln = _trunc_norm_int(mu, mu / 4, low, high)
        ln = min(ln, remain)
        core_lens.append(ln)
        remain -= ln
    return core_lens


def _avg_dif_length(total_len, avg_length):
    num_segments = total_len // avg_length
    if num_segments <= 1:
        return []

    lengths = [avg_length] * num_segments

    return lengths


def _calc_stdevs(seg: List[List[float]]) -> List[float]:
    """返回该子轨迹 7 个数值维度的标准差。"""
    stdevs = []
    for j in range(1, 8):
        col = [p[j] for p in seg]
        try:
            stdevs.append(statistics.stdev(col))
        except statistics.StatisticsError:
            stdevs.append(0.0)
            print("error in stddev calculation, using 0.0")
    return stdevs


def _add_noise(sub: List[List[float]],
               start: int,
               length: int,
               std: List[float] = None,
               reverse: bool = False,
               min_ratio: float = 0.1,
               max_ratio: float = 40.0) -> None:
    if length <= 0:
        return
    if std is None:
        stdevs = _calc_stdevs(sub)
    else:
        stdevs = std
    for i in range(length):
        pos = i if not reverse else (length - 1 - i)  # 0 → length-1
        w = (pos) / (length - 1) if length > 1 else 0  # 0→1
        ratio = min_ratio * ((max_ratio / min_ratio) ** w)  # 线性插值

        for j in range(1, 8):
            sigma = (stdevs[j - 1] or 1e-9) * ratio  # stdev=0 时给极小值
            sub[start + i][j] += random.gauss(0, 3 * sigma)


def single_traj_split(
        traj: List[List[float]],
        avg_length: int = 13,
        fix_overlap=False,
        std=None
) -> List[Tuple[List[float], List[List[float]], List[List[float]]]]:
    """
    返回 [(mid_point, sub_i, sub_{i+1}), ...] 列表。
    """

    new_traj = []
    for p in traj:
        new_traj.append(p[:8])
    traj = new_traj
    n = len(traj)
    if n < 16:
        return []
    if fix_overlap:
        return fix_overlap_length_single_traj_split(traj, avg_length, std=std, overlap=2)
    else:
        raise Exception("single_traj_split only support fix_overlap=True now")
    # 1) 先按“核心”长度切分
    core_lens = _sample_core_lengths(n, avg_length)
    core_starts, core_ends = [0], []  # 核心起止（end 为 exclusive）
    for ln in core_lens:
        core_ends.append(core_starts[-1] + ln)
        core_starts.append(core_ends[-1])
    core_starts.pop()  # 与 ends 对齐

    # 2) 给每段预扩 +5 / -5（最大重叠半径）
    ext_starts = [max(0, s - 3) for s in core_starts]
    ext_ends = [min(n, e + 3) for e in core_ends]

    # 3) 构建最终 pair
    pairs = []
    for i in range(len(core_ends) - 1):
        k = core_ends[i]  # 切分基准索引
        overlap = random.randint(-3, 3)  # [-5,5]

        if overlap >= 0:  # -------- 正重叠 --------
            r = overlap
            a_start = ext_starts[i]
            a_end = k + r + 1  # +1 因 python slice end-exclusive
            b_start = k - r
            b_end = ext_ends[i + 1]

            sub_a = copy.deepcopy(traj[a_start:a_end])
            sub_b = copy.deepcopy(traj[b_start:b_end])

            # 噪声：对称重复区 [k-r, k+r]
            dup_len = 2 * r + 1
            _add_noise(sub_a, len(sub_a) - dup_len, dup_len, std=std, reverse=False)  # A 尾部
            _add_noise(sub_b, 0, dup_len, std=std, reverse=True)  # B 头部

        else:  # -------- 间隙 --------
            g = -overlap
            a_start = ext_starts[i]
            a_end = k - g  # 不含缺失区
            b_start = k + g
            b_end = ext_ends[i + 1]

            # 若间隙过大可能导致空段，这里做保护
            if a_end <= a_start or b_end <= b_start:
                continue

            sub_a = copy.deepcopy(traj[a_start:a_end])
            sub_b = copy.deepcopy(traj[b_start:b_end])
            # 间隙不加噪声

        mid_point = copy.deepcopy(traj[k])  # 永远取原轨迹的 k 点

        # 长度兜底：依然 >0 且 ≤40
        if 10 <= len(sub_a) <= 40 and 10 <= len(sub_b) <= 40:
            pairs.append((mid_point, sub_a, sub_b))

    return pairs


def fix_overlap_length_single_traj_split(traj, avg_length, std, overlap):
    n = len(traj)
    assert overlap % 2 == 0, "overlap must be even"

    # 1) 先按“核心”长度切分
    core_lens = _avg_dif_length(n, avg_length)
    core_starts, core_ends = [0], []  # 核心起止（end 为 exclusive）
    for ln in core_lens:
        core_ends.append(core_starts[-1] + ln)
        core_starts.append(core_ends[-1])
    core_starts.pop()  # 与 ends 对齐

    # 2) 构建最终 pair
    pairs = []
    for i in range(len(core_ends) - 1):
        k = core_ends[i]  # 切分基准索引
        # overlap = random.randint(-3, 3)         # [-5,5]

        # -------- 正重叠 --------
        r = overlap // 2
        a_start = core_starts[i]
        a_end = k - r
        b_start = k + r
        b_end = core_ends[i + 1]

        sub_a = copy.deepcopy(traj[a_start:a_end])
        sub_b = copy.deepcopy(traj[b_start:b_end])
        fuse_a = copy.deepcopy(traj[a_start:a_end + 2 * r])
        fuse_b = copy.deepcopy(traj[b_start - 2 * r:b_end])
        label_a = copy.deepcopy(traj[a_start:a_end + 2 * r])
        label_b = copy.deepcopy(traj[b_start - 2 * r:b_end])
        assert len(label_a) == 12
        assert len(label_b) == 12
        dup_len = 2
        f_dup_len = 4
        _add_noise(sub_a, len(sub_a) - dup_len, dup_len, std=std, reverse=False)  # A 尾部
        _add_noise(sub_b, 0, dup_len, std=std, reverse=True)  # B 头部
        _add_noise(fuse_a, len(fuse_a) - f_dup_len, f_dup_len, std=std, reverse=False)
        _add_noise(fuse_b, 0, f_dup_len, std=std, reverse=True)

        mid_point = copy.deepcopy(traj[k])  # 永远取原轨迹的 k 点

        # 长度兜底：依然 >0 且 ≤40
        if 10 <= len(sub_a) <= 40 and 10 <= len(sub_b) <= 40:
            # pairs.append((mid_point, sub_a, sub_b, (label_a, label_b),(fuse_a, fuse_b)))
            pairs.append((mid_point, fuse_a[:10], fuse_b[-10:], (label_a, label_b), (fuse_a, fuse_b)))


    return pairs

if __name__ == "__main__":
    pass
