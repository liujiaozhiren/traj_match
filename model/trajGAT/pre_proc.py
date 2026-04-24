import numpy as np

import cfg
from model.handler import ModelHandler
from model.trajGAT.loader import TrajGraphDataLoader
from model.trajGAT.model import GraphTransformer
from model.trajGAT.tree_build import build_qtree, get_pre_embedding
from model.util import cmb_pre_post_traj, reshape_pairs


def strip_unused_info(traj_pairs):
    new_pairs = []
    for traj_pair in traj_pairs:
        _, pre, post = traj_pair
        new_pre, new_post = [], []
        for point in pre:
            new_pre.append((point[1], point[2]))
        for point in post:
            new_post.append((point[1], point[2]))
        new_pairs.append([new_pre, new_post])
    return new_pairs




def trajGAT_preproc(traj_pair_all, traj_pair_train, traj_pair_valid, traj_pair_test):

    traj_pair_all = strip_unused_info(traj_pair_all)
    traj_pair_train = strip_unused_info(traj_pair_train)
    traj_pair_valid = strip_unused_info(traj_pair_valid)
    traj_pair_test = strip_unused_info(traj_pair_test)

    traj_all = cmb_pre_post_traj(traj_pair_all)
    # traj_pair_train = reshape_pairs(traj_pair_train)
    # traj_pair_valid = reshape_pairs(traj_pair_valid)
    # traj_pair_test = reshape_pairs(traj_pair_test)

    stats = traj_statistics(traj_all)
    qtree = build_qtree(traj_all, stats["x_range"], stats["y_range"], max_items=50, max_depth=50)
    qtree_name2id, pre_embedding = get_pre_embedding(qtree, cfg.dim)
    model = ModelHandler(model_name='trajGAT',embedding=pre_embedding).to(cfg.device)
    traj_pair_train = TrajGraphDataLoader(traj_pair_train, qtree, qtree_name2id,d_lap_pos=8,num_workers=0,traj_stats=stats).get_data_loader()
    traj_pair_valid = TrajGraphDataLoader(traj_pair_valid, qtree, qtree_name2id,d_lap_pos=8,num_workers=0,traj_stats=stats).get_data_loader()
    traj_pair_test = TrajGraphDataLoader(traj_pair_test, qtree, qtree_name2id,d_lap_pos=8,num_workers=0,traj_stats=stats).get_data_loader()
    return model, traj_pair_train, traj_pair_valid, traj_pair_test

def traj_statistics(traj_data):
    # 1) 收集所有经纬度
    lats, lons = [], []
    for traj in traj_data:
        for p in traj:  # 依次展开点
            lon, lat = p[0], p[1]
            assert len(p)==2
            lats.append(lat)
            lons.append(lon)
    lats = np.asarray(lats, dtype=float)
    lons = np.asarray(lons, dtype=float)
    # 2) 计算统计量
    lat_min, lat_max = lats.min(), lats.max()
    lon_min, lon_max = lons.min(), lons.max()
    x_range, lon_mean, lon_std = (lon_min, lon_max), lons.mean(), lons.std()
    y_range, lat_mean, lat_std = (lat_min, lat_max), lats.mean(), lats.std()
    data_features = (lon_mean, lon_std, lat_mean, lat_std)
    stats = {"x_range": x_range, "y_range": y_range, "data_features": data_features, }
    return stats



def data_construct(traj_data, max_items=50, max_depth=50):
    # (lon_mean, lon_std, lat_mean, lat_std) x_range y_range
    stats = traj_statistics(traj_data)
    qtree = build_qtree(traj_data, stats["x_range"], stats["y_range"], max_items, max_depth)
    qtree_name2id, pre_embedding = get_pre_embedding(qtree, cfg.dim)
    # embedding -> model
    model = GraphTransformer(pre_embedding)
    # qtree, qtree_name2id -> data_loader
    data_loader = TrajGraphDataLoader(traj_data, qtree, qtree_name2id,d_lap_pos=8,num_workers=0,traj_stats= stats)
    # model_optim = optim.Adam(model.parameters(), lr=cfg.lr)
    return
