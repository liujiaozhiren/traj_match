import pandas as pd
import numpy as np
import math

# ---------------- 辅助：大圆距离（米） ---------------- #
def haversine(lon1, lat1, lon2, lat2, R=6371000):
    lon1, lat1, lon2, lat2 = map(math.radians, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))

# ----------------- 轨迹→ID 映射器 ----------------- #
class TrajMapper:
    def __init__(self, k: int = 6):
        self.k = k
        self.coord2id, self.id2coord = {}, {}

    def traj_to_id_seq(self, traj):
        return [self._assign_id(pt[1], pt[2]) for pt in traj]

    def traj_to_time_seq(self, traj):
        return [pt[0] for pt in traj]

    # ---------- 内部 ---------- #
    def _assign_id(self, lon, lat):
        key = (round(lon, self.k), round(lat, self.k))
        if key not in self.coord2id:
            new_id = len(self.coord2id)          # 连续自增
            self.coord2id[key] = new_id
            self.id2coord[new_id] = key
        return self.coord2id[key]

# ----------------- 构图主函数 ----------------- #
def build_graph_from_trajs(traj_list, k: int = 6):
    """
    返回
    ----
    df_edge  : edge_id, s_node, e_node, dist
    df_node  : node_id, lng, lat
    edge_idx : ndarray(E,2)
    mapper   : TrajMapper
    """
    mapper, edge_records = TrajMapper(k), []

    for traj in traj_list:
        ids = mapper.traj_to_id_seq(traj)
        for (p1, p2), (id1, id2) in zip(zip(traj, traj[1:]), zip(ids, ids[1:])):
            if id1 == id2:           # 同一点连续出现可忽略
                continue
            lon1, lat1 = p1[1:3]
            lon2, lat2 = p2[1:3]
            d = haversine(lon1, lat1, lon2, lat2)
            edge_records.append((id1, id2, d))

    # --- 去重并编号 edge_id --- #
    edge_df = (pd.DataFrame(edge_records, columns=["s_node", "e_node", "dist"])
                 .drop_duplicates(ignore_index=True))
    edge_df.insert(0, "edge_id", range(len(edge_df)))   # (edge_id, s, e, d)

    # --- 点表 --- #
    node_ids, coords = zip(*sorted(mapper.id2coord.items()))
    lngs, lats = zip(*coords)
    node_df = pd.DataFrame({"node_id": node_ids, "lng": lngs, "lat": lats})

    edge_index = edge_df[["s_node", "e_node"]].to_numpy(dtype=int)
    return edge_df, node_df, edge_index, mapper