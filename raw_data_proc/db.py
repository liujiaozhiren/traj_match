from typing import List, Tuple

import cfg
# from model.train import model_train
from raw_data_proc.search import analyse_pairs, sample_db, trajectory_density

shenzhen = [1746680000000, 114.057868, 22.543099, 500]  # t= 2025 年 5 月 8 日 12 : 53 : 20 shenzhen lon lat，
if cfg.use_delta:
    shenzhen = [50000, 0, 0, 0]

# ---------- 类型别名 ----------
Point  = List[float]                               # 原 8 维点
RPoint = List[float]                               # 仅保留 t,lat,lon,alt 4 维
Pair   = Tuple[Point, List[Point], List[Point]]    # (mid, A, B)
CleanPair = Tuple[RPoint, List[RPoint], List[RPoint]]

# ---------- 1. 只取 0/1/2/3 号分量 ----------
def _keep_t_lat_lon_alt(pt: Point) -> RPoint:
    if cfg.input_height:
        return [pt[0], pt[1], pt[2], pt[3]]
    else:
        return [pt[0], pt[1], pt[2]]
# ---------- 2. 轨迹点减 mid_point ----------
def _normalize_traj(traj: List[RPoint], mid: RPoint) -> List[RPoint]:
    if cfg.input_height:
        return [[p[0] - mid[0]+shenzhen[0],          # Δt
                 p[1] - mid[1]+shenzhen[1],          # Δlat
                 p[2] - mid[2]+shenzhen[2],          # Δlon
                 p[3] - mid[3]+shenzhen[3]]          # Δalt
                for p in traj]
    else:
        return [[p[0] - mid[0]+shenzhen[0],          # Δt
                 p[1] - mid[1]+shenzhen[1],          # Δlat
                 p[2] - mid[2]+shenzhen[2]]          # Δlon
                for p in traj]

# ---------- 3. 主清洗函数 ----------
def clean_pairs(pairs: List[Pair]) -> List[CleanPair]:
    cleaned = []
    for item in pairs:
        mid, A, B, label, fuse =item[0], item[1], item[2], item[3], item[4]
        if len(A) < 3 or len(B) < 3:
            continue
        L_A = [_keep_t_lat_lon_alt(p) for p in label[0]]
        L_B = [_keep_t_lat_lon_alt(p) for p in label[1]]
        F_A = [_keep_t_lat_lon_alt(p) for p in fuse[0]]
        F_B = [_keep_t_lat_lon_alt(p) for p in fuse[1]]
        mid_4   = _keep_t_lat_lon_alt(mid)
        A_4     = [_keep_t_lat_lon_alt(p) for p in A]
        B_4     = [_keep_t_lat_lon_alt(p) for p in B]
        A_norm  = _normalize_traj(A_4, mid_4)
        B_norm  = _normalize_traj(B_4, mid_4)
        L_A_norm = _normalize_traj(L_A, mid_4)
        L_B_norm = _normalize_traj(L_B, mid_4)
        F_A_norm = _normalize_traj(F_A, mid_4)
        F_B_norm = _normalize_traj(F_B, mid_4)
        cleaned.append((mid_4, A_norm, B_norm, L_A_norm, L_B_norm, F_A_norm, F_B_norm))
    return cleaned

def get_dim_max_min(pairs):
    dim0max, dim0min = float('-inf'), float('inf')
    dim1max, dim1min = float('-inf'), float('inf')
    dim2max, dim2min = float('-inf'), float('inf')
    dim3max, dim3min = float('-inf'), float('inf')

    for mid, A, B in pairs:
        for p in A + B:
            dim0max = max(dim0max, p[0])
            dim0min = min(dim0min, p[0])
            dim1max = max(dim1max, p[1])
            dim1min = min(dim1min, p[1])
            dim2max = max(dim2max, p[2])
            dim2min = min(dim2min, p[2])
            dim3max = max(dim3max, p[3])
            dim3min = min(dim3min, p[3])
    print(f"dim0: max={dim0max}, min={dim0min}, "
          f"dim1: max={dim1max}, min={dim1min}, "
          f"dim2: max={dim2max}, min={dim2min}, "
          f"dim3: max={dim3max}, min={dim3min}")

def get_new_db_pairs(single_pairs, multi_pairs):
    pairs = single_pairs + multi_pairs
    new_pairs = clean_pairs(pairs)
    # get_dim_max_min(new_pairs)
    db = sample_db(new_pairs, ratio=0.02, seed=42)
    density = trajectory_density(db)

    # learning part
    train_db = [p for p in new_pairs if p not in db]
    print(f"all pair{len(new_pairs)}  train/v_pair:{len(train_db)}, test pair:{len(db)} ")

    acc = model_train(train_db, db, all=new_pairs, model_name='mydiff')
    # learning式精确度 model_name 已支持
    #     'traj2simvec'
    #     'trajGAT'
    #     'ST2Vec'
    #     'pred_lstm'
    acc = model_train(train_db, db, all=new_pairs, model_name='pred_lstm')
    print(len(db), density, "dtw", acc)

    # 平均距离精确度
    acc = analyse_pairs(db, method='rule')
    print(len(db), density, "rule", acc)
    # exit(0)

    # 规则式相似度精确度
    # time_attr None 无时间惩罚 method linear 线性时间惩罚 method my 多项式时间惩罚 times多项式次数
    attr = {'method':'my','xishu':0.000043,'times':5}
    acc = analyse_pairs(db, method='dtw',time_attr=attr)
    print(attr,acc)
    attr = {'method':'my','xishu':0.0000428,'times':5}
    acc = analyse_pairs(db, method='dtw',time_attr=attr)
    print(attr, acc)
    attr = {'method':'my','xishu':0.00004127,'times':5}
    acc = analyse_pairs(db, method='dtw',time_attr=attr)
    print(attr, acc)

    acc = model_train(train_db, db, all=new_pairs, model_name='diff')


    # acc2 = analyse_pairs(db, method='hausdorff')
    # print(len(db),density,"hausdorff",acc2)
    return None
