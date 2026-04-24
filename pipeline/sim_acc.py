import pickle

import cfg
from model.train import model_train
from pipeline.proc_traj_pair import preproc_mul_p
from raw_data_proc.db import clean_pairs
from raw_data_proc.my_proc import proc_multi_single_traj
from raw_data_proc.search import sample_db, analyse_pairs, trajectory_density

if __name__ == "__main__":
    print("版本 v10.22.02.04")
    multi_traj_cmb, single_traj_cmb = pickle.load(open(cfg.con_file, 'rb'))
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    mul_p = preproc_mul_p(mul_p)
    pairs = mul_p + sing_p
    all = clean_pairs(pairs)
    # get_dim_max_min(new_pairs)
    db = sample_db(all, ratio=0.0047, seed=42)
    density = trajectory_density(db)
    train_db = [p for p in all if p not in db]
    print(f"all pair{len(all)}  train/v_pair:{len(train_db)}, test pair:{len(db)} , density:{density}")
    acc = analyse_pairs(db, method='dtw', time_attr=None)
    print(f'dtw acc: {acc}')
    acc = analyse_pairs(db, method='hausdorff', time_attr=None)
    print(f'hausdorff acc: {acc}')

    # model_name=
        # 'traj2simvec'
        # 'trajGAT'
        # 'ST2Vec'
        # 'pred_lstm'
    # acc = model_train(train_db, db, all=all, model_name='ST2Vec')
    # print(f'traj2simvec acc: {acc}')
