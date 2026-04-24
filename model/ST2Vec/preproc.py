import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

import cfg
from cfg import device
from model.ST2Vec.model.node2vec import read_graph, get_node_embedding
from model.ST2Vec.model.time2vec import proc_timeseq_embedding_nodeseq
from model.handler import ModelHandler
from model.util import cmb_pre_post_traj, reshape_pairs


def strip_mid_point(all_traj):
    ret = []
    for pairs in all_traj:
        _, pre, post = pairs[0],pairs[1],pairs[2]
        ret.append((pre, post)) # 只保留 pre 和 post 部分)
    return ret

def strip_all_mid_point(train_set, valid_set, db, all_traj):
    train_set = strip_mid_point(train_set)
    valid_set = strip_mid_point(valid_set)
    db = strip_mid_point(db)
    all_traj = strip_mid_point(all_traj)
    return train_set, valid_set, db, all_traj

def get_all_traj(all_traj):
    traj_all_tmp = cmb_pre_post_traj(all_traj)
    if len(traj_all_tmp) == 2: # "ST2Vec_preproc should have two parts: pre and post trajs"
        traj_all = []
        traj_all.extend(traj_all_tmp[0])
        traj_all.extend(traj_all_tmp[1])
        traj_all_tmp = traj_all
    return traj_all_tmp

def ST2Vec_preproc(train_set, valid_set, db, all_traj):
    train_set, valid_set, db, all_traj = strip_all_mid_point(train_set, valid_set, db, all_traj)
    all_traj = get_all_traj(all_traj)
    edge_index, num_node, mapper, edge_df = read_graph(all_traj)
    edge_index = torch.LongTensor(edge_index).t().contiguous().to(device)
    train_set = reshape_pairs(train_set)
    valid_set = reshape_pairs(valid_set)
    db = reshape_pairs(db)
    embeddings = get_node_embedding(edge_index, num_node, device=device)

    assert len(train_set) == 2, "ST2Vec_preproc should have two parts: pre and post trajs"
    train_pre, train_post = train_set[0], train_set[1]
    valid_pre, valid_post = valid_set[0], valid_set[1]
    db_pre, db_post = db[0], db[1]
    train_pre_ts_emb, train_pre_id_seq = proc_timeseq_embedding_nodeseq(mapper, train_pre)
    train_post_ts_emb, train_post_id_seq = proc_timeseq_embedding_nodeseq(mapper, train_post)
    valid_pre_ts_emb, valid_pre_id_seq = proc_timeseq_embedding_nodeseq(mapper, valid_pre)
    valid_post_ts_emb, valid_post_id_seq = proc_timeseq_embedding_nodeseq(mapper, valid_post)
    db_pre_ts_emb, db_pre_id_seq = proc_timeseq_embedding_nodeseq(mapper, db_pre)
    db_post_ts_emb, db_post_id_seq = proc_timeseq_embedding_nodeseq(mapper, db_post)

    train_data = list(zip(train_pre_ts_emb, train_pre_id_seq, train_post_ts_emb, train_post_id_seq))
    valid_data = list(zip(valid_pre_ts_emb, valid_pre_id_seq, valid_post_ts_emb, valid_post_id_seq))
    test_data = list(zip(db_pre_ts_emb, db_pre_id_seq, db_post_ts_emb, db_post_id_seq))

    dataloader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, collate_fn=loader_collate_fn)
    validloader = DataLoader(valid_data, batch_size=cfg.batch_size, shuffle=False, collate_fn=loader_collate_fn)
    testloader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=False, collate_fn=loader_collate_fn)

    edge_index, edge_attr = edge_df[["s_node", "e_node"]].to_numpy(), edge_df[["dist"]].to_numpy()
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    embeddings = torch.tensor(embeddings, dtype=torch.float)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    road_network = Data(x=embeddings, edge_index=edge_index, edge_attr=edge_attr)
    model = ModelHandler(model_name='ST2Vec', network=road_network).to(cfg.device)

    return model, dataloader, validloader, testloader
    # ts_emb, id_seq = proc_timeseq_embedding_nodeseq(mapper, traj_list)

def loader_collate_fn(batch):
    batch_size = len(batch)

    pre_ts_emb, pre_id_seq, post_ts_emb, post_id_seq = zip(*batch)
    # max_len = max(max(len(s) for s in pre_ts_emb), max(len(s) for s in post_ts_emb))
    # pre_ts_emb = pad_tensor_sequences_fixed(pre_ts_emb, max_len, pad_val=0.0, dtype=torch.float32, device=device)
    # pre_id_seq = pad_tensor_sequences_fixed(pre_id_seq, max_len, pad_val=0, dtype=torch.int32, device=device)
    # post_ts_emb = pad_tensor_sequences_fixed(post_ts_emb, max_len, pad_val=0.0, dtype=torch.float32, device=device)
    # post_id_seq = pad_tensor_sequences_fixed(post_id_seq, max_len, pad_val=0, dtype=torch.int32, device=device)

    return (pre_ts_emb, pre_id_seq, post_ts_emb, post_id_seq), None

def pad_tensor_sequences_fixed(
    seq_list,
    max_l: int,
    pad_val: float = 0.0,
    dtype=torch.float32,
    device=None,
):
    # ---- 统一 dtype / device ---- #
    seqs = [torch.as_tensor(s, dtype=dtype, device=device) for s in seq_list]
    B = len(seqs)
    feat_shape = seqs[0].shape[1:]          # 可能为空 tuple()

    # ---- 预分配全 pad_val ---- #
    padded_shape = (B, max_l, *feat_shape)
    padded = torch.full(padded_shape, pad_val, dtype=dtype, device=device)

    lengths = torch.empty(B, dtype=torch.long, device=device)

    # ---- 拷贝 / 截断 ---- #
    for i, s in enumerate(seqs):
        L = min(s.size(0), max_l)
        padded[(i, slice(0, L))] = s[:L]    # 适配任意 feat_shape
        lengths[i] = L

    mask = torch.arange(max_l, device=device).expand(B, -1) < lengths.unsqueeze(1)
    return padded, lengths, mask