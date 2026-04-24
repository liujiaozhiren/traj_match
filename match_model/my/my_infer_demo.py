import torch, dgl
from typing import List, Tuple
import torch.nn.functional as F
import cfg

from match_model.my.graph_fusion import GraphEmbedding
from match_model.my.mygraph import build_one_graph_dgl, build_pair_graph_dgl
from match_model.my.pairmatch import get_pair_map


def batch_embedding_graph(trajA_pre, trajA_post, trajB_pre, trajB_post, device='cpu'):
    # trajA_pre:  (B, k0, 2)
    # trajA_post: (B, n, k, 2)
    B = len(trajA_pre)
    pairs = []
    for i in range(B):
        gA = build_one_graph_dgl(trajA_pre[i], trajA_post[i], directed=True, add_reverse=False, add_edge_len=True, device=device)
        gB = build_one_graph_dgl(trajB_pre[i], trajB_post[i], directed=True, add_reverse=False, add_edge_len=True, device=device)

        pairs.append((gA, gB))

    graphs = []
    for gA, gB in pairs:
        graphs += [gA.to(device), gB.to(device)]
        bg = dgl.batch(graphs).to(device)
    # 对齐边特征名：将 'elen' 拷到 'ef'（若存在）
    if 'elen' in bg.edata and 'ef' not in bg.edata:
        bg.edata['ef'] = bg.edata['elen']
    return bg




def batch_pairs_graph(trajA_pre, trajA_post, trajB_pre, trajB_post, device='cpu'):
    # trajA_pre:  (B, k0, 2)
    # trajA_post: (B, n, k, 2)
    B = len(trajA_pre)
    pairs = []
    for i in range(B):
        gA = build_one_graph_dgl(trajA_pre[i], trajA_post[i], directed=True, add_reverse=False, add_edge_len=True, device=device)
        gB = build_one_graph_dgl(trajB_pre[i], trajB_post[i], directed=True, add_reverse=False, add_edge_len=True, device=device)
        pair_map = get_pair_map(trajA_post[i], trajB_post[i])
        merged_g = build_pair_graph_dgl(gA, gB, k0=trajA_pre.size(1), n=trajA_post.size(1), k=trajA_post.size(2), t_gap=cfg.diff_infer_len+1, device=device, pair_map=pair_map)
        pairs.append(merged_g)
        if cfg.device == 'cpu' and False:  # 仅CPU时可视化
            from match_model.my.draw import plot_merged_graph_intensity
            plot_merged_graph_intensity(merged_g, k0, n, k)

    # graphs = []

    bg = dgl.batch(pairs).to(device)
    # 对齐边特征名：将 'elen' 拷到 'ef'（若存在）
    if 'elen' in bg.edata and 'ef' not in bg.edata:
        bg.edata['ef'] = bg.edata['elen']
    return bg

def embedding_loss(logits, labels, dim=128):
    logits = logits.view(-1,2,dim) # B*2*D
    # 计算B*2之间的距离
    g0, g1 = logits[:, 0], logits[:, 1]  # (B, D) each
    # 计算g0和g1之间的距离
    dist = torch.norm(g0 - g1, p=2, dim=1)  # (B,)
    # 这里label也是B类型的bool 代表dist距离是否应该为0，如果是1那对应位置的dist应该是0 反之应该很大
    loss = torch.mean((labels.float() * dist**2) + ((1 - labels).float() * torch.clamp(1.0 - dist, min=0.0)**2))
    return loss

def pair_loss(logits, labels):
    # labels 可能是 bool，转成 float；其余不改
    if labels.dtype != torch.float32 and labels.dtype != torch.float16 and labels.dtype != torch.bfloat16:
        labels = labels.float()

    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')  # 形状与 logits 相同

    # 保证返回 (B,)
    if loss.dim() == 1:              # (B,)
        return loss
    else:                            # (B,1) 或 (B,D)
        return loss.mean(dim=1)      # 按最后一维做均值 -> (B,)


def generate_claw_trajs(B, k0, n, k, gap=1.0, branch_len=1.0, flip_B=True):
    """
    生成类似“爪子”的轨迹数据（用于可视化调试）

    参数：
        B: 批大小
        k0: 主干长度
        n: 分支数量
        k: 每个分支的节点数
        gap: 主干节点间的纵向距离
        branch_len: 每条分支的横向总长度
        flip_B: 若为 True，则 B 图是 A 图的垂直镜像（方向相反）

    返回：
        trajA_pre, trajA_post, trajB_pre, trajB_post
    """
    trajA_pre, trajA_post, trajB_pre, trajB_post = [], [], [], []
    import math
    for _ in range(B):
        # === A 图 ===
        # 主干：沿 y 轴上升（0, 0）→(0, (k0-1)*gap)
        k0_seq_A = torch.stack([torch.zeros(k0), torch.arange(k0) * gap+ gap], dim=1)

        # 分支：从主干顶端发出 n 条
        branches_A = []
        for j in range(n):
            angle = (-1)**j * (j+1) * 0.3  # 左右交替发散
            # 生成一条分支：从主干末端出发，沿 angle 方向延伸
            base = k0_seq_A[-1]

            dx = torch.linspace(0, branch_len * math.cos(angle), k+1)
            dy = torch.linspace(0, branch_len * math.sin(angle), k+1)
            branch = torch.stack([base[0] + dx[1:], base[1] + dy[1:]], dim=1)
            branches_A.append(branch)
        branches_A = torch.stack(branches_A, dim=0)

        # === B 图 ===
        if flip_B:
            # 镜像：沿 y 轴翻转并下移，使主干方向反转
            k0_seq_B = torch.stack([torch.zeros(k0), -torch.arange(k0) * gap], dim=1)
        else:
            k0_seq_B = k0_seq_A.clone()

        branches_B = []
        for j in range(n):
            angle = (-1)**j * (j+1) * 0.3
            base = k0_seq_B[-1]
            dx = torch.linspace(0, branch_len * math.cos(angle), k+1)
            dy = torch.linspace(0, branch_len * math.sin(angle), k+1)
            branch = torch.stack([base[0] + dx[1:], base[1] + dy[1:]], dim=1)
            branches_B.append(branch)
        branches_B = torch.stack(branches_B, dim=0)

        trajA_pre.append(k0_seq_A)
        trajA_post.append(branches_A)
        trajB_pre.append(k0_seq_B)
        trajB_post.append(branches_B)

    return (torch.stack(trajA_pre, dim=0),
            torch.stack(trajA_post, dim=0),
            torch.stack(trajB_pre, dim=0),
            torch.stack(trajB_post, dim=0))



# ===== demo：随机构造几对样本图，跑一遍推理（无标签） =====
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 你真实的输入应该是两套序列（每对两张图都各自有 k0 和 branches）。
    # 这里用随机数据示例：构造 B 对 (gA, gB)，每张图由主链 + n 条分支组成。
    from math import floor
    B, n, k0, k = 4, 3, 3, 2
    dim = 128
    # 构建模型（如有 checkpoint，取消注释加载）
    model = GraphEmbedding(node_in=2, use_edge_feat=True, rep_dim=dim).to(device)

    # 推理（无标签）
    model.eval()
    with torch.no_grad():
        # trajA_pre = torch.randn(B, k0, 2)
        # trajA_post = torch.randn(B, n, k, 2)
        # trajB_pre = torch.randn(B, k0, 2)
        # trajB_post = torch.randn(B, n, k, 2)

        trajA_pre, trajA_post, trajB_pre, trajB_post = generate_claw_trajs(B, k0, n, k)

        label = torch.randint(0,2,(B,)).to(device)
        # embedding
        bg_pairs = batch_embedding_graph(trajA_pre, trajA_post, trajB_pre, trajB_post, device=device)
        ret = model(bg_pairs, pairs=False)               # [B]
        loss = embedding_loss(ret, label, dim=dim)
        # or using pairs
        bg_pairs = batch_pairs_graph(trajA_pre, trajA_post, trajB_pre, trajB_post, device=device)
        ret = model(bg_pairs, pairs=True)               # [B]
        loss = pair_loss(ret, label)

        # probs = torch.sigmoid(ret)          # [B]，两图匹配为“正”的概率

    # # 打印结果
    # for i, p in enumerate(probs.tolist(), 1):
    #     print(f'Pair #{i}: match prob = {p:.3f}')