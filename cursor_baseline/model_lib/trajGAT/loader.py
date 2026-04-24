import copy
import math

import dgl
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

import scipy.sparse as sp

import cfg


class TrajDataset(Dataset):
    def __init__(self, traj_data, **kwargs):
        self.traj_data = traj_data  # [[pre, post], [pre, post], ...]

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.traj_data)

    def __getitem__(self, idx):
        return self.traj_data[idx]


class TrajGraphDataLoader:
    def __init__(self, traj_data, qtree, qtree_name2id, d_lap_pos, num_workers, traj_stats, is_shuffle=False) -> None:
        self.traj_data = traj_data
        self.d_lap_pos = d_lap_pos
        self.data_features = traj_stats["data_features"]# (lon_mean, lon_std, lat_mean, lat_std)
        self.num_workers = num_workers
        self.x_range = traj_stats["x_range"]
        self.y_range = traj_stats["y_range"]
        self.qtree = qtree
        self.qtree_name2id = qtree_name2id
        self.is_shuffle = is_shuffle

    def get_data_loader(self):
        self.dataset = TrajDataset(traj_data=self.traj_data)
        data_loader = DataLoader(self.dataset, batch_size=cfg.batch_size, shuffle=self.is_shuffle, num_workers=0, collate_fn=self._collate_func)
        # else:
        #     data_loader = DataLoader(self.dataset, batch_size=cfg.batch_size, shuffle=self.is_shuffle, num_workers=self.num_workers, collate_fn=self._collate_func)

        return data_loader

    def _collate_func(self, traj_pair_list):
        pres, posts = [], []
        for traj_pair in traj_pair_list:
            pre, post = traj_pair
            pres.append(pre)
            posts.append(post)
        traj_pair_list = [pres, posts]
        trajdict_list_list = self._prepare(traj_pair_list)
        trajgraph_list_list = self._build_graph(trajdict_list_list)

        return trajgraph_list_list, None

    def _prepare(self, traj_l_l):
        trajdict_pair_list = []
        for traj_l in traj_l_l:
            temp_list = []
            for traj in traj_l:
                padding_traj = []
                point_ids = []
                adj, add_feat, tree_id = self._build_adj_matrix_tree(traj)
                padding_traj.extend([(tp[0], tp[1], 0, 0) for tp in traj])
                padding_traj.extend(add_feat)

                point_ids.extend([0 for _ in range(len(traj))])
                point_ids.extend(tree_id)

                # 用来标注 真实轨迹点 和 虚拟节点的 feature , 1: truth node    0: visual node
                flag = torch.zeros((len(padding_traj), 1))
                flag[0 : len(traj)] = 1

                temp_list.append({"traj": self._normalize(padding_traj), "adj": adj, "flag": flag, "point_ids": torch.tensor(point_ids).long()})
            trajdict_pair_list.append(temp_list)

        return trajdict_pair_list

    def _normalize(self, traj):
        lon_mean, lon_std, lat_mean, lat_std = self.data_features
        traj = torch.tensor(traj)
        if traj.shape[1] == 2:
            traj = traj - torch.tensor([lon_mean, lat_mean])
            traj = traj * torch.tensor([1 / lon_std, 1 / lat_std])
        elif traj.shape[1] == 4:
            traj = traj - torch.tensor([lon_mean, lat_mean, 0, 0])
            traj = traj * torch.tensor([1 / lon_std, 1 / lat_std, 1, 1])

        return traj

    def _build_adj_matrix_label(self, traj):
        def _get_grid_id(point):
            num_x_grid = math.ceil((self.x_range[1] - self.x_range[0]) / self.tail_delta)

            x, y = point
            x_grid = (x - self.x_range[0]) // self.tail_delta
            y_grid = (y - self.y_range[0]) // self.tail_delta
            grid_index = int(y_grid * num_x_grid + x_grid)

            return (x_grid, y_grid), grid_index

        traj_len = len(traj)
        u = []
        v = []
        edge_data = []

        for row_idx, row_point in enumerate(traj):
            _, row_grid_id = _get_grid_id(row_point)
            for col_idx, col_point in enumerate(traj):
                _, col_grid_id = _get_grid_id(col_point)
                if row_grid_id != col_grid_id:
                    u.append(row_idx)
                    v.append(col_idx)
                    edge_data.append(1)

        u = np.array(u)
        v = np.array(v)
        edge_data = np.array(edge_data)
        adj_matrix = sp.coo_matrix((edge_data, (u, v)), shape=(traj_len, traj_len))

        return adj_matrix

    def _build_adj_matrix_tree(self, traj, vir_node_layers=1):
        if vir_node_layers == 1:
            # 根据qtree结构，构建涉及 vir_node_layer 层父节点的全连接graph
            traj_len = len(traj)
            point2treel = []  # 每个点在qtree中对应的 vir_node_layer 个父节点

            for t_point in traj:
                t_list = self.qtree.intersect(t_point, method="tree")
                point2treel.append(t_list)

            node_num = traj_len
            tree_set = []
            for treel in point2treel:
                tree_set.extend(treel)
            tree_set = set(tree_set)
            node_num += len(tree_set)

            id_start = traj_len
            tree2id = {}
            center_wh_feat = []
            tree_id = []
            for tt in tree_set:
                tree2id[id(tt)] = id_start
                id_start += 1
                this_x, this_y = tt.center
                this_w, this_h = tt.width, tt.height
                center_wh_feat.append((this_x, this_y, this_w, this_h))  # 将格子 中心点坐标 宽 高 作为他的特征

                if self.qtree_name2id:
                    # 将每个结点对应的id进行存储
                    tree_id.append(self.qtree_name2id[id(tt)])
                else:
                    # 不用word embedding的时候，补0
                    tree_id.append(0)

            u = []
            v = []
            edge_data = []

            #  连接图上纵向的边
            for point_index in range(traj_len):
                tree_list = point2treel[point_index]
                # 连接 point —— 叶子树
                for tt in tree_list:
                    u.append(point_index)
                    v.append(tree2id[id(tt)])
                    edge_data.append(1)

                    u.append(tree2id[id(tt)])
                    v.append(point_index)
                    edge_data.append(1)

            # 连接图上横向的边
            tree_ids = list(tree2id.values())
            for i in range(len(tree_ids) - 1):
                for j in range(i + 1, len(tree_ids)):
                    u.append(tree_ids[i])
                    v.append(tree_ids[j])
                    edge_data.append(1)

                    u.append(tree_ids[j])
                    v.append(tree_ids[i])
                    edge_data.append(1)

            # 自身的连边
            for this_id in range(node_num):
                u.append(this_id)
                v.append(this_id)
                edge_data.append(1)

            u = np.array(u)
            v = np.array(v)
            edge_data = np.array(edge_data)
            adj_matrix = sp.coo_matrix((edge_data, (u, v)), shape=(node_num, node_num))
        else:
            # 根据qtree结构，构建涉及 vir_node_layer 层父节点的全连接graph
            traj_len = len(traj)
            point2treel = []  # 每个点在qtree中对应的 vir_node_layer 个父节点

            for t_point in traj:
                t_list = self.qtree.intersect(t_point, method="all_tree")
                point2treel.append([i[1] for i in t_list[-1 : -1 - vir_node_layers : -1]])

            node_num = traj_len
            tree_set = []
            for treel in point2treel:
                tree_set.extend(treel)
            tree_set = set(tree_set)
            node_num += len(tree_set)

            id_start = traj_len
            tree2id = {}
            center_wh_feat = []
            tree_id = []
            for tt in tree_set:
                tree2id[id(tt)] = id_start
                id_start += 1
                this_x, this_y = tt.center
                this_w, this_h = tt.width, tt.height
                center_wh_feat.append((this_x, this_y, this_w, this_h))  # 将格子 中心点坐标 宽 高 作为他的特征

                if self.qtree_name2id:
                    # 将每个结点对应的id进行存储
                    tree_id.append(self.qtree_name2id[id(tt)])
                else:
                    # 不用word embedding的时候，补0
                    tree_id.append(0)

            u = []
            v = []
            edge_data = []

            #  连接图上纵向的边
            for point_index in range(traj_len):
                tree_list = point2treel[point_index]
                # 连接 point —— 叶子树
                u.append(tree2id[id(tree_list[0])])
                v.append(point_index)
                edge_data.append(1)

                v.append(tree2id[id(tree_list[0])])
                u.append(point_index)
                edge_data.append(1)

                # 连接 叶子树 —— 更高层的树
                for tt in range(1, len(tree_list)):
                    u.append(tree2id[id(tree_list[tt])])
                    v.append(tree2id[id(tree_list[tt - 1])])
                    edge_data.append(1)

                    u.append(tree2id[id(tree_list[tt - 1])])
                    v.append(tree2id[id(tree_list[tt])])
                    edge_data.append(1)

            # 连接图上横向的边
            tree_ids = set()
            for jj in range(traj_len):
                tree_ids.add(point2treel[jj][-1])
            tree_ids = list(tree_ids)

            for i in range(len(tree_ids) - 1):
                for j in range(i + 1, len(tree_ids)):
                    u.append(tree2id[id(tree_ids[i])])
                    v.append(tree2id[id(tree_ids[j])])
                    edge_data.append(1)

                    u.append(tree2id[id(tree_ids[j])])
                    v.append(tree2id[id(tree_ids[i])])
                    edge_data.append(1)

            # 自身的连边
            for this_id in range(node_num):
                u.append(this_id)
                v.append(this_id)
                edge_data.append(1)

            u = np.array(u)
            v = np.array(v)
            edge_data = np.array(edge_data)
            try:
                adj_matrix = sp.coo_matrix((edge_data, (u, v)), shape=(node_num, node_num))
            except:
                print("edge_data:\n", edge_data)
                print("U\n", u)
                print("V\n", v)
                print("NN\n", node_num)
                print("FINISH")
        return adj_matrix, center_wh_feat, tree_id

    def _build_graph(self, trajdict_l_l):
        trajgraph_list_list = []

        for trajdict_l in trajdict_l_l:
            trajgraph_l = []
            for trajdict in trajdict_l:
                node_features = trajdict["traj"].float()
                id_features = trajdict["point_ids"]
                adj = trajdict["adj"]
                flag = trajdict["flag"]

                # Create the DGL Graph
                g = dgl.from_scipy(adj, eweight_name="feat")

                padding_node_num = g.num_nodes() - node_features.shape[0]
                padding_node = torch.zeros((padding_node_num, node_features.shape[1])).float()
                all_node_features = torch.cat((node_features, padding_node), dim=0)

                g.ndata["feat"] = all_node_features
                g.ndata["flag"] = flag
                g.ndata["id"] = id_features

                g = laplacian_positional_encoding(g, self.d_lap_pos)

                trajgraph_l.append(g)

            trajgraph_list_list.append(copy.deepcopy(trajgraph_l))

        return trajgraph_list_list



def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    from packaging import version

    def get_adj(g, fmt="csr"):
        ver = version.parse(dgl.__version__)
        # DGL 2.x: no scipy_fmt on adj(); build scipy from internal SparseMatrix / COO.
        if ver >= version.parse("2.0.0"):
            sm = g.adj()
            r, c = sm.coo()
            v = sm.val
            n = int(g.num_nodes())
            if sm.nnz == 0:
                return sp.csr_matrix((n, n))
            A = sp.coo_matrix(
                (v.cpu().numpy(), (r.cpu().numpy(), c.cpu().numpy())),
                shape=(n, n),
            )
            return A.asformat(fmt) if fmt != "coo" else A
        if ver >= version.parse("0.6.0"):
            return g.adj(scipy_fmt=fmt)
        return g.adjacency_matrix_scipy(return_edge_ids=False).asformat(fmt)

    A = get_adj(g).astype(float)

    #A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])

    # # if temp.shape[1] != pos_enc_dim:
    # #     print("!!!! ERROR", temp.shape)
    # g.ndata["lap_pos_feat"] = torch.from_numpy(EigVec[:, 1 : pos_enc_dim + 1]).float()

    # ---------- 3) 取前 k 列 + 填充 ----------
    n = g.number_of_nodes()
    k = min(pos_enc_dim, n - 1)  # 至多 n-1 个非平凡特征向量
    vecs = EigVec[:, 1: k + 1]  # (n, k)

    if k < pos_enc_dim:  # 若列数不足 → 右侧补 0
        pad = np.zeros((n, pos_enc_dim - k))
        vecs = np.concatenate([vecs, pad], axis=1)  # (n, pos_enc_dim)

    g.ndata["lap_pos_feat"] = torch.from_numpy(vecs).float()

    return g
