from match_model.my.my_infer_demo import pair_loss
from pipeline.load import hydra_graph
import cfg

def downstream_fn(match, info_pre, info_post, sel_pre, sel_post, label=None):
    if cfg.rule == 'match':
        ret = match_down(match, info_pre, info_post, sel_pre, sel_post)
    elif cfg.rule == 'sim':
        ret = match_sim(info_pre, info_post, sel_pre, sel_post)
    if label is not None:
        loss_vec = pair_loss(ret, label)
        return loss_vec
    return ret

def match_down(match, info_pre, info_post, sel_pre, sel_post):
    bg_pairs = hydra_graph(info_pre, info_post, sel_pre, sel_post, pair=True)
    ret = match(bg_pairs, pairs=True)
    print('sjn', info_pre[0][0])
    print('sjn', info_post[0][0])
    print('sjn', sel_pre[0][0])
    print('sjn', sel_post[0][0])
    return ret


def match_sim(info_pre, info_post, sel_pre, sel_post):
    B = info_pre.shape[0]
    print('sjn', info_pre[0][0])
    exit(0)
    return ret

def dtw(dist):
    import numpy as np
    m, n = dist.shape
    D = np.full((m + 1, n + 1), np.inf, dtype=float)
    D[0, 0] = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            D[i, j] = dist[i - 1, j - 1] + min(D[i - 1, j],
                                             D[i, j - 1],
                                             D[i - 1, j - 1])
    return float(D[m, n])