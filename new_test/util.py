from pretrain.code.util import clip_traj_eq_length, equal_gap_traj
import numpy as np
import torch

def raw_traj_clip_preproc(trajs, length=50):
    ret_trajs = []
    for traj in trajs:
        # traj = trajs[traj_key]
        part_trajs = clip_traj_eq_length(traj, length)
        for traj in part_trajs:
            if equal_gap_traj(traj):
                traj = relocate_traj(traj)
                if should_keep_traj_const_blocks(traj):
                    ret_trajs.append(traj)
    return ret_trajs

def should_keep_traj_const_blocks(traj, eps=1e-3, max_blocks=10):
    arr = np.asarray(traj, dtype=float)
    if len(arr) <= 1:
        return True  # 太短直接丢
    # 相邻是否发生变化
    change = np.any(np.abs(np.diff(arr, axis=0)) > eps, axis=1)  # (T-1,)
    blocks = 1 + int(change.sum())  # 连续常量块个数
    return blocks >= max_blocks

default_shenzhen = [114.057868, 22.543099]

def relocate_traj(traj):
    lons = [p[1] for p in traj]
    lats = [p[2] for p in traj]
    mean_lon = sum(lons) / len(lons)
    mean_lat = sum(lats) / len(lats)

    # new_traj = []
    xy_traj = []
    for p in traj:
        # new_traj.append([p[1]-mean_lon+default_shenzhen[0], p[2]-mean_lat+default_shenzhen[1]])
        x_, y_ = p[1]-mean_lon+default_shenzhen[0], p[2]-mean_lat+default_shenzhen[1]
        xy_traj.append(lonlat_to_xy_at_origin(x_, y_, default_shenzhen))
        # 往着下面加 你是傻逼么 别tm 改上面的
    return xy_traj
from math import cos, radians

R = 6378137.0  # WGS-84

def lonlat_to_xy_at_origin(lon: float, lat: float, origin):
    """把经纬度(lon,lat)换算成以 origin=(lon0,lat0) 为(0,0)的平面坐标(米)。x向东，y向北。"""
    lon0, lat0 = origin
    kx = R * cos(radians(lat0))         # 经度→米（在 origin 纬度处）
    dx = kx * radians(lon - lon0)
    dy = R  * radians(lat - lat0)
    return dx, dy

