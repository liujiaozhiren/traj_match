import pickle

from zte.dist_cal import pre_proc, pairwise_traj_distance_np, greedy_assignment

if __name__ == '__main__':

    assign = 'greedy'  # or 'hungarian'
    trajlistA, trajlistB = [], []
    fn = '2980.pkl' # or 16.pkl
    height = True if fn == '16.pkl' else False
    pairs = pickle.load(open('2980.pkl', 'rb'))

    # pairs =[(A,B),...] 中每个样本代表配对的轨迹，

    for i in range(len(pairs)):
        trajA = pairs[i][0]
        trajB = pairs[i][1]
        if fn == '2980.pkl': # for 2980.pkl 没有ps值 默认置为0
            trajA = [[point[0], point[1], point[2],0] for point in trajA]
            trajB = [[point[0], point[1], point[2],0] for point in trajB]
        trajlistA.append(trajA)
        trajlistB.append(trajB)

    trajlistA = pre_proc(trajlistA, height=height)
    trajlistB = pre_proc(trajlistB, height=height)
    traj_dist_matrix = pairwise_traj_distance_np(trajlistA, trajlistB, height=height)
    # 因此正确的矩阵值应是对角线最小。
    if assign == 'greedy':
        match_for_A, total_cost = greedy_assignment(traj_dist_matrix)
    # 正确的匹配值就是 i->i