import math
from pathlib import Path
import os
import glob
import pandas as pd

from new_match.traj_dist_my import trajectory_common_distance
import os

from raw_data_proc.search import dist


#os.sys.path.append('../../')  # 添加上级目录到路径



def get_trajs(df, pre=False):
    trajectories = []
    for gid, g in df.groupby('global_id', sort=False):
        traj = g[['timer_ts', 'src_sensing_carrier_id', 'ps', 'Lat', 'Lon', 'Height',
                  'speed', 'gcs_vx', 'gcs_vy', 'gcs_vz']].values.tolist()
        traj.sort(key=lambda x: x[0])
        for idx in range(len(traj)):
            if pre:
                traj[idx][0] = traj[idx][0] #- 350400.0 - 1750332000000  # 将时间戳转换为相对时间
            else:
                traj[idx][0] = traj[idx][0] #- 711600.0 - 1750332000000
        if len(traj) > 10:
            trajectories.append(traj)
    gnb_dict = {}
    for traj in trajectories:
        gnb_id = traj[0][1]  # 获取第一个点的 gnb_id
        for point in traj:
            tmp_gnb_id = point[1]
            if pre:
                assert tmp_gnb_id == gnb_id, f"Mismatch in gnb_id: {tmp_gnb_id} != {gnb_id}"
        if gnb_id not in gnb_dict:
            gnb_dict[gnb_id] = []
        gnb_dict[gnb_id].append(traj)

    return trajectories, gnb_dict


def pinjie(path):
    # ==== 1. 拼接分段 CSV ====
    csv_files = sorted(glob.glob(os.path.join(path, '*.csv')))  # 按文件名排序

    dfs = []
    header = None
    for i, f in enumerate(csv_files):
        if i == 0:
            df = pd.read_csv(f)  # 第一段带列名
            header = df.columns  # 记录列名
        else:
            df = pd.read_csv(f, header=None, names=header)  # 其余段无列名
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    return full_df


def get_owner(trajs):
    owner = {}
    for traj in trajs:
        this_owner = {}
        for point in traj:
            tmp_owner = point[1]
            if tmp_owner not in this_owner:
                this_owner[tmp_owner] = 1
            else:
                this_owner[tmp_owner] += 1
        if len(this_owner) == 1:
            if traj[0][1] not in owner:
                owner[traj[0][1]] = []
            owner[traj[0][1]].append(traj)
        elif len(this_owner) == 2:
            # get all keys
            keys = list(this_owner.keys())
            # sort keys
            sorted_keys = sorted(keys, reverse=True)
            if str(sorted_keys) not in owner:
                owner[str(sorted_keys)] = []
            owner[str(sorted_keys)].append(traj)
        else:
            if 'all' not in owner:
                owner['all'] = []
            owner['all'].append(traj)
    return owner


def get_501_502(traj):
    traj_501 = []
    traj_502 = []
    last = 501.0
    assert last == traj[0][1]
    for point in traj:
        gnb = point[1]
        if last == gnb == 501.0:
            traj_501.append(point)
        elif last == 501.0 and gnb == 502.0:
            traj_502.append(point)
        elif last == 502.0 and gnb == 501.0:
            raise Exception("501 and 502 should not be in the same trajectory")
        elif last == gnb == 502.0:
            traj_502.append(point)
        else:
            raise Exception(f"Unexpected gnb_id: {gnb} at point {point}")

    return traj_501, traj_502


def get_matched_traj(sub_traj, trajs):
    d_list, n_list = [], []
    for i, traj in enumerate(trajs):
        n, d, _ = trajectory_common_distance(sub_traj, traj)
        if n == 0:
            continue
        if n * 1.0 / len(sub_traj) < 0.8:
            continue
        d_list.append(d)
        n_list.append(n)
    # get 3 top large d 's index
    top_indices = sorted(range(len(d_list)), key=lambda i: d_list[i], reverse=False)[:5]
    ddd = [d_list[i] for i in top_indices]
    return list(zip(top_indices, ddd, n_list))


def find_sub_traj(traj, trajs_501, trajs_502):
    sub_501, sub_502 = get_501_502(traj)
    ret_501 = get_matched_traj(sub_501, trajs_501)
    ret_502 = get_matched_traj(sub_502, trajs_502)
    return ret_501, ret_502

def static(a,b,c):
    all = []
    all.extend(a)
    all.extend(b)
    all.extend(c)
    # -------------- 收集所有相对量 -----------------
    dlat, dlon, dalt, mid_lats = [], [], [], []
    for traj in all:
        # mid_lats.append(mid[1])            # mid_point 的绝对纬度 (°)
        for p in traj:
            dlat.append(p[3])              # 已经是 “Δlat(°)”
            dlon.append(p[4])              # “Δlon(°)”
            dalt.append(p[5])              # “Δalt(m)”

    # -------------- 近似 ° → m --------------------
    lat_span_deg = max(dlat) - min(dlat)
    lon_span_deg = max(dlon) - min(dlon)
    alt_span_m   = max(dalt) - min(dalt)

    # 参考纬度：所有 mid_point 纬度平均
    ref_lat = 22.543099  # 默认深圳纬度
    lat_rad = math.radians(ref_lat)

    dy = lat_span_deg * 111_132.0                 # Δnorth (m)
    dx = lon_span_deg * 111_320.0 * math.cos(lat_rad)  # Δeast  (m)
    dz = alt_span_m                               # Δalt    (m)

    # 若某方向跨度为 0 → ε
    eps = 1e-6
    vol = max(dy, eps) * max(dx, eps) * max(dz, eps)
    square_meters = dy * dx  # 面积（m²）
    return len(all) / vol, len(all) / square_meters

def abc():

    path = './'
    # file
    traj_dict = {}

    for file_path in ['./before', './after']:
        df = pinjie(file_path)
        traj_dict[file_path] = get_trajs(df, file_path == './before')
    #print(1)

    a = traj_dict['./before'][1][501.0]
    b = traj_dict['./before'][1][502.0]
    c = traj_dict['./before'][1][503.0]
    aft = traj_dict['./after'][0]
    aft_owner = get_owner(aft)
    fused = aft_owner['[502.0, 501.0]']

    for traj in fused:
        ar, br = find_sub_traj(traj, a, b)
        #print(f" 501 ret:{ar} \t\t|| 502 ret:{br}")
    #print(traj_dict)
    [11, 7, 10, 12, 2, 3, 9, 13, 15, 4, 1, 0, 8, 6, 14, 5]
    [11, 7, 10, 12, 2, 3, 9, 13, 15, 4, 1, 0, 8, 6, 14, 5]

    all = []
    ret = static(a, b, c)
    print(f"static 密度(n/m3){ret[0]}  密度(n/m2){ret[1]} ")
    find_traj_list = []
    find_traj_list.extend(b)
    find_traj_list.extend(c)
    cnt = 0
    error, match_dist = [],[]
    for idx in range(16):
        traj = a[idx]
        dist_list = []
        for find_idx, find_traj in enumerate(find_traj_list):
            _, dist_, _ = trajectory_common_distance(traj,find_traj)
            dist_list.append(dist_)
        sorted_indices = sorted(range(len(dist_list)), key=lambda i: dist_list[i])
        top = sorted_indices[0]
        match_dist.append(dist_list[top])
        if top == idx:
            cnt +=1
        else:
            error.append(idx)
    print(f"Matched trajectories count: {cnt},{error},{match_dist}")

    error, match_dist = [],[]
    cnt = 0
    for idx in range(16):
        traj = a[idx]
        dist_list = []
        for find_idx, find_traj in enumerate(find_traj_list):
            tmp_traj = [[point[0], point[3], point[4], point[5]] for point in traj]
            tmp_find_traj = [[point[0], point[3], point[4], point[5]] for point in find_traj]
            dist_ = dist(tmp_traj,tmp_find_traj, 'dtw', time_attr={'method':'linear'})
            dist_list.append(dist_)
        sorted_indices = sorted(range(len(dist_list)), key=lambda i: dist_list[i])
        top = sorted_indices[0]
        match_dist.append(dist_list[top])
        if top == idx:
            cnt +=1
        else:
            error.append(idx)
    print(f"Matched trajectories count dtw: {cnt},{error},{match_dist}")

    error, match_dist = [], []
    cnt = 0
    for idx in range(16):
        traj = a[idx]
        dist_list = []
        for find_idx, find_traj in enumerate(find_traj_list):
            tmp_traj = [[point[0], point[3], point[4], point[5]] for point in traj]
            tmp_find_traj = [[point[0], point[3], point[4], point[5]] for point in find_traj]
            dist_ = dist(tmp_traj,tmp_find_traj, 'hausdorff')
            dist_list.append(dist_)
        sorted_indices = sorted(range(len(dist_list)), key=lambda i: dist_list[i])
        top = sorted_indices[0]
        match_dist.append(dist_list[top])
        if top == idx:
            cnt +=1
        else:
            error.append(idx)
    print(f"Matched trajectories count hausdorff: {cnt},{error},{match_dist}")


abc()

# base_dir = Path(path).resolve()  # 根目录（一级目录）
# for root, _, files in os.walk(base_dir):
#     root_path = Path(root)
#     # 计算相对深度：base_dir 本身深度为 0，子目录深度为 1
#     depth = len(root_path.relative_to(base_dir).parts)
#     for name in files:
#         file_path = root_path / name
#         if name == 'before':
#             df = pinjie(file_path)
#             traj_dict[name] = get_trajs(df)
#         elif name == 'after':
#             df = pinjie(file_path)
#             traj_dict[name] = get_trajs(df)
#
# print(traj_dict)
