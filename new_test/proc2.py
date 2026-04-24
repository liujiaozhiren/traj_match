import cfg
import torch
from torch.utils.data import DataLoader
def preproc_pairdb2XYtrajs(db_pairs):
    trajs = []
    for item in db_pairs:
        mid, A, B, L_A, L_B = item[0], item[1], item[2], item[3], item[4]
        L_B = L_B[::-1]
        L_A, L_B = relocate_traj(L_A, L_B)
        item = [L_A, L_B]
        trajs.append(item)
    #trajs = [trajs_pre, trajs_post]
    trajs_tensor = torch.tensor(trajs, dtype=torch.float32)
    trajs_tensor = trajs_tensor.view(len(db_pairs), 2, cfg.diff_pre_len+ cfg.diff_infer_len, 2).to(cfg.device)
    return trajs_tensor

def gen_con_pretrain_XY_loader(train_db, db, shuffle=True):
    train_db_raw_traj = preproc_pairdb2XYtrajs(train_db)
    db_raw_traj = preproc_pairdb2XYtrajs(db)
    if cfg.device == 'cpu':
        train_db_raw_traj = train_db_raw_traj[:100]
        db_raw_traj = db_raw_traj[:100]
    dataloader = DataLoader(train_db_raw_traj, batch_size=200, shuffle=shuffle, num_workers=0)
    validloader = DataLoader(db_raw_traj, batch_size=20, shuffle=False, num_workers=0)
    return dataloader, validloader




default_shenzhen = [114.057868, 22.543099]

def relocate_traj(L_A, L_B):

    lons = [p[1] for p in L_A] + [p[1] for p in L_B]
    lats = [p[2] for p in L_A] + [p[2] for p in L_B]
    mean_lon = sum(lons) / len(lons)
    mean_lat = sum(lats) / len(lats)

    xy_L_A, xy_L_B = [], []
    for p in L_A:
        x_, y_ = p[1]-mean_lon+default_shenzhen[0], p[2]-mean_lat+default_shenzhen[1]
        # x_, y_ = p[1], p[2]
        xy_L_A.append(lonlat_to_xy_at_origin(x_, y_, default_shenzhen))
        # xy_L_A.append((x_,y_))
    for p in L_B:
        x_, y_ = p[1]-mean_lon+default_shenzhen[0], p[2]-mean_lat+default_shenzhen[1]
        # x_, y_ = p[1], p[2]
        xy_L_B.append(lonlat_to_xy_at_origin(x_, y_, default_shenzhen))
        # xy_L_B.append((x_,y_))
    return xy_L_A, xy_L_B

from math import cos, radians

R = 6378137.0  # WGS-84

def lonlat_to_xy_at_origin(lon: float, lat: float, origin):
    """把经纬度(lon,lat)换算成以 origin=(lon0,lat0) 为(0,0)的平面坐标(米)。x向东，y向北。"""
    lon0, lat0 = origin
    kx = R * cos(radians(lat0))         # 经度→米（在 origin 纬度处）
    dx = kx * radians(lon - lon0)
    dy = R  * radians(lat - lat0)
    return dx, dy