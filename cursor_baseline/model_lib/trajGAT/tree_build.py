import torch

from cursor_baseline.model_lib.trajGAT.node2vec.node2vec import node2vec_embed
from cursor_baseline.model_lib.trajGAT.qtree import Index, get_qtree_feat


# def pre_proc(traj_pair_all, traj_pair_train, traj_pair_valid, traj_pair_test):
#     stats = traj_statistics(traj_pair_all)
#     pres, posts = [], []
#     for traj_pair in traj_pair_train:
#         _, pre, post = traj_pair
#         pre_traj, post_traj = [],[]
#         for point in pre:
#             pre_traj.append((point[1], point[2]))  # (lon, lat)
#         for point in post:
#             post_traj.append((point[1], point[2]))
#
#
# def data_construct(traj_data, max_items=50, max_depth=50):
#     # (lon_mean, lon_std, lat_mean, lat_std) x_range y_range
#     stats = traj_statistics(traj_data)
#     qtree = build_qtree(traj_data, stats["x_range"], stats["y_range"], max_items, max_depth)
#     qtree_name2id, pre_embedding = get_pre_embedding(qtree, cfg.dim)
#     # embedding -> model
#     model = GraphTransformer(pre_embedding)
#     # qtree, qtree_name2id -> data_loader
#     data_loader = TrajGraphDataLoader(traj_data, qtree, qtree_name2id,d_lap_pos=8,num_workers=0,traj_stats= stats)
#     # model_optim = optim.Adam(model.parameters(), lr=cfg.lr)
#     return


def build_qtree(traj_data, x_range, y_range, max_items, max_depth):
    qtree = Index(bbox=(x_range[0], y_range[0], x_range[1], y_range[1]), max_items=max_items, max_depth=max_depth)
    point_num = 0
    #print("Building Q-Tree...")
    for traj in traj_data:
        for point in traj:
            point_num += 1
            x, y = point[0], point[1]  # 假设point是一个包含经纬度的元组或列表
            qtree.insert(point_num, (x, y, x, y))

    # print("traj point nums:", point_num)

    return qtree


def get_pre_embedding(qtree, d_model):
    vir_id_edge_list, vir_id2center, word_embedding_name2id = get_qtree_feat(qtree)

    # print("Edge number used in node2vec:", len(vir_id_edge_list))
    # print("Point number used in node2vec:", len(vir_id2center))

    vir_pre_embedding = node2vec_embed(vir_id_edge_list, d_model)
    vir_pre_embedding = torch.tensor(vir_pre_embedding, dtype=torch.float)
    vir_name2id = word_embedding_name2id

    # 对预训练得到的embedding进行归一化 min-max
    # print(vir_pre_embedding.min(), vir_pre_embedding.max())
    vir_min = vir_pre_embedding.min(axis=0)[0]
    vir_max = vir_pre_embedding.max(axis=0)[0]
    vir_pre_embedding = (vir_pre_embedding - vir_min) / (vir_max - vir_min)
    #print(vir_pre_embedding.min(), vir_pre_embedding.max())

    # 添加全零的embedding
    vir_pre_embedding = torch.cat([torch.zeros(1, vir_pre_embedding.shape[1]), vir_pre_embedding], dim=0)

    # print("The number of word embedding:", vir_pre_embedding.shape)

    return vir_name2id, vir_pre_embedding
