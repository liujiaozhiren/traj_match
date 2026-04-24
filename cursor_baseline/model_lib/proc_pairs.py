"""Vendored from new_test/proc.py + raw_data_proc.my_proc.get_stdevs (no repo-root deps)."""

from __future__ import annotations

import copy
import random
import statistics

default_shenzhen = [0, 114.057868, 22.543099]


def get_stdevs(trajs):
    """Per-dimension stdev over dimensions 1..7 across all points in all trajs."""
    seg = []
    for traj in trajs:
        seg.extend(traj)
    stdevs = []
    for j in range(1, 8):
        col = [p[j] for p in seg]
        try:
            stdevs.append(statistics.stdev(col))
        except statistics.StatisticsError:
            stdevs.append(0.0)
    return stdevs


def _add_noise(
    sub,
    start: int,
    length: int,
    std=None,
    reverse: bool = False,
    min_ratio: float = 0.1,
    max_ratio: float = 40.0,
) -> None:
    if length <= 0:
        return
    if std is None:
        raise ValueError("std must be provided")
    stdevs = std
    for i in range(length):
        pos = i if not reverse else (length - 1 - i)
        w = (pos) / (length - 1) if length > 1 else 0
        ratio = min_ratio * ((max_ratio / min_ratio) ** w)

        for j in range(1, 8):
            sigma = (stdevs[j - 1] or 1e-9) * ratio
            sub[start + i][j] += random.gauss(0, 3 * sigma)


def _normalize_traj(traj, mid, input_height=False):
    if input_height:
        return [
            [
                p[0] - mid[0] + default_shenzhen[0],
                p[1] - mid[1] + default_shenzhen[1],
                p[2] - mid[2] + default_shenzhen[2],
                p[3] - mid[3] + default_shenzhen[3],
            ]
            for p in traj
        ]
    return [
        [p[0] - mid[0] + default_shenzhen[0], p[1] - mid[1] + default_shenzhen[1], p[2] - mid[2] + default_shenzhen[2]]
        for p in traj
    ]


def _keep_t_lat_lon_alt(pt, input_height=False):
    if input_height:
        return [pt[0], pt[1], pt[2], pt[3]]
    return [pt[0], pt[1], pt[2]]


def clean_pairs(pairs):
    cleaned = []
    for item in pairs:
        mid, A, B, label, fuse = item[0], item[1], item[2], item[3], item[4]
        if len(A) < 3 or len(B) < 3:
            continue
        L_A = [_keep_t_lat_lon_alt(p) for p in label[0]]
        L_B = [_keep_t_lat_lon_alt(p) for p in label[1]]
        F_A = [_keep_t_lat_lon_alt(p) for p in fuse[0]]
        F_B = [_keep_t_lat_lon_alt(p) for p in fuse[1]]
        mid_4 = _keep_t_lat_lon_alt(mid)
        A_4 = [_keep_t_lat_lon_alt(p) for p in A]
        B_4 = [_keep_t_lat_lon_alt(p) for p in B]
        A_norm = _normalize_traj(A_4, mid_4)
        B_norm = _normalize_traj(B_4, mid_4)
        L_A_norm = _normalize_traj(L_A, mid_4)
        L_B_norm = _normalize_traj(L_B, mid_4)
        F_A_norm = _normalize_traj(F_A, mid_4)
        F_B_norm = _normalize_traj(F_B, mid_4)
        cleaned.append((mid_4, A_norm, B_norm, L_A_norm, L_B_norm, F_A_norm, F_B_norm))
    return cleaned


def _avg_dif_length(total_len, avg_length):
    num_segments = total_len // avg_length
    if num_segments <= 1:
        return []
    return [avg_length] * num_segments


def fix_overlap_length_single_traj_split(traj, avg_length, std, overlap):
    n = len(traj)
    assert overlap % 2 == 0, "overlap must be even"

    core_lens = _avg_dif_length(n, avg_length)
    core_starts, core_ends = [0], []
    for ln in core_lens:
        core_ends.append(core_starts[-1] + ln)
        core_starts.append(core_ends[-1])
    core_starts.pop()

    pairs = []
    for i in range(len(core_ends) - 1):
        k = core_ends[i]
        r = overlap // 2
        a_start = core_starts[i]
        a_end = k - r
        b_start = k + r
        b_end = core_ends[i + 1]

        sub_a = copy.deepcopy(traj[a_start:a_end])
        sub_b = copy.deepcopy(traj[b_start:b_end])
        fuse_a = copy.deepcopy(traj[a_start : a_end + 2 * r])
        fuse_b = copy.deepcopy(traj[b_start - 2 * r : b_end])
        label_a = copy.deepcopy(traj[a_start : a_end + 2 * r])
        label_b = copy.deepcopy(traj[b_start - 2 * r : b_end])
        assert len(label_a) == 12
        assert len(label_b) == 12
        dup_len = 2
        f_dup_len = 4
        _add_noise(sub_a, len(sub_a) - dup_len, dup_len, std=std, reverse=False)
        _add_noise(sub_b, 0, dup_len, std=std, reverse=True)
        _add_noise(fuse_a, len(fuse_a) - f_dup_len, f_dup_len, std=std, reverse=False)
        _add_noise(fuse_b, 0, f_dup_len, std=std, reverse=True)

        mid_point = copy.deepcopy(traj[k])

        if 10 <= len(sub_a) <= 40 and 10 <= len(sub_b) <= 40:
            pairs.append((mid_point, fuse_a[:10], fuse_b[-10:], (label_a, label_b), (fuse_a, fuse_b)))

    return pairs


def proc_multi_single_traj(multi_traj_cmb, single_traj_cmb):
    mul_p, sing_p = [], []
    stdevs = get_stdevs(single_traj_cmb)
    for traj in single_traj_cmb:
        aaa = fix_overlap_length_single_traj_split(traj, 11, std=stdevs, overlap=2)
        sing_p.extend(aaa)
    return mul_p, sing_p
