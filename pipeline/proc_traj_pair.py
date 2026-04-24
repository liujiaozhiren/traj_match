import torch,cfg
from torch.utils.data import DataLoader

def preproc_mul_p(pairs):
    return []

def preproc_pairdb2trajs(db_pairs):
    trajs = []
    for item in db_pairs:
        mid, A, B, L_A, L_B = item[0], item[1], item[2], item[3], item[4]
        L_B = L_B[::-1]
        item = [L_A, L_B]
        trajs.append(item)
    #trajs = [trajs_pre, trajs_post]
    trajs_tensor = torch.tensor(trajs, dtype=torch.float32)
    trajs_tensor = trajs_tensor.view(len(db_pairs), 2, cfg.diff_pre_len+ cfg.diff_infer_len, 3).to(cfg.device)
    return trajs_tensor

def preproc_pairdb2trajs4FT(db_pairs):
    trajs = []
    for item in db_pairs:
        mid, A, B, L_A, L_B = item[0], item[1], item[2], item[3], item[4]
        B = B[::-1]
        item = [A, B]
        trajs.append(item)
    #t rajs = [trajs_pre, trajs_post]
    trajs_tensor = torch.tensor(trajs, dtype=torch.float32)
    trajs_tensor = trajs_tensor.view(len(db_pairs), 2, cfg.diff_pre_len, 3).to(cfg.device)
    return trajs_tensor


def gen_con_pretrain_loader(train_db, db, shuffle=True):
    train_db_raw_traj = preproc_pairdb2trajs(train_db)
    db_raw_traj = preproc_pairdb2trajs(db)
    if cfg.device == 'cpu':
        train_db_raw_traj = train_db_raw_traj[:10]
        db_raw_traj = db_raw_traj[:10]
    dataloader = DataLoader(train_db_raw_traj, batch_size=cfg.small_pt_batchsize, shuffle=shuffle, num_workers=0)
    validloader = DataLoader(db_raw_traj, batch_size=cfg.small_pt_batchsize//10, shuffle=False, num_workers=0)
    return dataloader, validloader


def gen_ft_loader(train_db, db, shuffle=True):
    train_db_raw_traj = preproc_pairdb2trajs4FT(train_db)
    db_raw_traj = preproc_pairdb2trajs4FT(db)
    if cfg.device == 'cpu':
        train_db_raw_traj = train_db_raw_traj[:10]
        db_raw_traj = db_raw_traj[:10]
    dataloader = DataLoader(train_db_raw_traj, batch_size=cfg.pre_gen_ft_batchsize, shuffle=shuffle, num_workers=0)
    validloader = DataLoader(db_raw_traj, batch_size=cfg.pre_gen_ft_batchsize, shuffle=False, num_workers=0)
    return dataloader, validloader, db_raw_traj[:,1]