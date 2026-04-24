import pickle

import torch

import cfg
from match_model.my.graph_fusion import GraphEmbedding
from pipeline.diff import finetune_train2
from pipeline.diff_train import prepare_Diff_Traj_pretrain, load_pretrain_model, pretrain_parallel
from pipeline.proc_traj_pair import preproc_mul_p
from pipeline.rl_train import rl2_train
from raw_data_proc.db import clean_pairs
from raw_data_proc.my_proc import proc_multi_single_traj
from raw_data_proc.search import sample_db

if __name__ == "__main__":
    print("版本 v10.22.02.04")
    # 预训练
    if not cfg.skip_pretrain:
        ## 预训练数据读取
        model, data = prepare_Diff_Traj_pretrain()
        preDiff, postDiff = model
        dataloader, validloader = data
        ## 并行/串行预训练
        # model_fn = pretrain(trainloader=dataloader, validloader=validloader, preDiff=preDiff, postDiff=postDiff)
        model_fn = pretrain_parallel(trainloader=dataloader, validloader=validloader, preDiff=preDiff, postDiff=postDiff, continue_train=True)
        print('Pretrain finished, model files:', model_fn)
        preDiff.load_state_dict(torch.load(model_fn[0], map_location=cfg.device))
        postDiff.load_state_dict(torch.load(model_fn[1], map_location=cfg.device))
    else:
        preDiff, postDiff = load_pretrain_model()
    match = GraphEmbedding(node_in=2, use_edge_feat=True, rep_dim=cfg.match_dim).to(cfg.device)

    # pair 数据读取
    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, 'rb'))
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    mul_p = preproc_mul_p(mul_p)
    pairs = mul_p + sing_p
    all = clean_pairs(pairs)
    # get_dim_max_min(new_pairs)
    db = sample_db(all, ratio=0.1, seed=42)
    # density = trajectory_density(db)
    train_db = [p for p in all if p not in db]
    print(f"all pair{len(all)}  train/v_pair:{len(train_db)}, test pair:{len(db)} ")

    train_db = train_db[:105]
    db = db[:105]

    # 继续训练
    # model_fn = con_pretrain(train_db, db, preDiff=preDiff, postDiff=postDiff)
    # print(f'finish continue pretrain, start matching fine-tune')

    # 微调
    match.load_state_dict(torch.load("model_match_0.6698762035763411_trajdiff1024.pth"), strict=True)

    preDiff.unet.load_state_dict(torch.load("pretrain/file/model_para_diff_pre_con322.67313766_trajdiff1024.pth"), strict=True)
    postDiff.unet.load_state_dict(torch.load("pretrain/file/model_para_diff_post_con322.67313766_trajdiff1024.pth"), strict=True)
    # diff_train, diff_db = get_diff_data(preDiff, postDiff, train_db, db)

    # finetune_train2(train_db, db, preDiff=preDiff, postDiff=postDiff, match=match)

    rl2_train(train_db, db, preDiff=preDiff, postDiff=postDiff, match=match)