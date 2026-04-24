import os
import pickle
import statistics
from pathlib import Path

import os, torch, torch.multiprocessing as mp
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
try: mp.set_start_method("spawn", force=True)
except RuntimeError: pass
torch.set_num_threads(1)


from raw_data_proc.db import get_new_db_pairs
from raw_data_proc.traj_clip import multi_traj_match_fetch, single_traj_split

def proc_raw_file():
    if not os.path.exists('traj_cmb.pkl'):
        from file_parse import parse_pcap
        path = '../raw-data'
        # file
        traj_dict = {}
        base_dir = Path(path).resolve()  # 根目录（一级目录）
        for root, _, files in os.walk(base_dir):
            root_path = Path(root)
            # 计算相对深度：base_dir 本身深度为 0，子目录深度为 1
            depth = len(root_path.relative_to(base_dir).parts)

            for name in files:
                file_path = root_path / name
                if depth == 0:  # 一级目录中的文件
                    print(f"Processing file1: {file_path}")
                    key = name.split('.')[0]  # 假设文件名格式为 key.ext
                    if not (str(file_path).endswith('.pcapng') or str(file_path).endswith('.pcap')):
                        continue
                    ret = parse_pcap(file_path)
                    traj_dict[key] = ret
                    # f2(file_path)
                elif depth == 1:  # 二级目录中的文件
                    # f1(file_path)
                    print(f"Processing file2: {file_path}")
                    key = root_path.name
                    if key not in traj_dict:
                        traj_dict[key] = []
                    if not (str(file_path).endswith('.pcapng') or str(file_path).endswith('.pcap')):
                        continue
                    ret = parse_pcap(file_path)
                    traj_dict[key].append(ret)
                else:
                    raise Exception("Unsupported directory depth: only 1 or 2 levels are allowed.")
        # print(traj_dict)
        traj_cmb = {}
        for k, v in traj_dict.items():
            if type(v) == list:
                a, b, c = [], [], {}
                for item in v:
                    aa, bb, cc = item
                    a.extend(aa)
                    b.extend(bb)
                    c.update(cc)
                traj_cmb[k] = (a, b, c)
            else:
                traj_cmb[k] = v
        multi_traj_cmb = {}
        single_traj_cmb = []
        for k, v in traj_cmb.items():
            a, b, c = v
            single_traj_cmb.extend(b)
            multi_traj_cmb.update(c)
        # cnt=0

        pickle.dump((multi_traj_cmb, single_traj_cmb), open('traj_cmb.pkl', 'wb'))
    else:
        multi_traj_cmb, single_traj_cmb = pickle.load(open('traj_cmb.pkl', 'rb'))
    return multi_traj_cmb, single_traj_cmb
    #pickle.dump(traj_pairs, open('traj_pairs.pkl', 'wb'))

def post_proc_mul_traj(mul_p):
    new_mul_p = []
    for mid, A, B in mul_p:
        if len(A) < 10 or len(B) < 10:
            continue
        if A[0][0] > B[0][0]:
            continue
        _A = A[-11:-1]
        _B = B[1:11]
        labelA = A[-11:]+[B[0]]
        labelB = [A[-1]]+B[:11]
        label = (labelA, labelB)
        new_mul_p.append((mid, _A, _B, label))
    return new_mul_p

def get_stdevs(trajs):
    """返回该子轨迹 7 个数值维度的标准差。"""
    # seg 是把所有trajs里的轨迹都 extend
    seg = []
    for traj in trajs:
        seg.extend(traj)
    stdevs = []
    for j in range(1, 8):
        col = [p[j] for p in seg]
        try:
            stdevs.append(statistics.stdev(col))
        except statistics.StatisticsError:
            stdevs.append(0.0)
            print("error in stddev calculation, using 0.0")
    return stdevs

def proc_multi_single_traj(multi_traj_cmb, single_traj_cmb):
    mul_p, sing_p = [], []
    for k, v in multi_traj_cmb.items():
        aaa = multi_traj_match_fetch(k, v)
        mul_p.extend(aaa)
        # cnt+=len(aaa)
    mul_p = post_proc_mul_traj(mul_p)
    # 统计标准差
    stdevs = get_stdevs(single_traj_cmb)
    for traj in single_traj_cmb:
        aaa = single_traj_split(traj, 11 , True, std=stdevs)
        sing_p.extend(aaa)
    return mul_p, sing_p

if __name__ == '__main__':

    multi_traj_cmb, single_traj_cmb = proc_raw_file()
    mul_p, sing_p = proc_multi_single_traj(multi_traj_cmb, single_traj_cmb)
    db_traj = get_new_db_pairs(sing_p, mul_p)
    # db_traj = mul_p + sing_p
    #
    # print(f"Total trajectories: {len(db_traj)}")
    # one = next(iter(multi_traj_cmb.items()))
    # multi_traj_match_fetch(one)
    # print(traj_cmb)
