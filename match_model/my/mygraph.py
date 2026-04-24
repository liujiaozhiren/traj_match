import torch
import dgl

def build_pair_graph_dgl(gA, gB, k0, n, k, t_gap=0, device='cpu', ab_bidir=True, pair_map=None):
    """
    将 gA(主干->末端) 与 gB(末端->主干) 合成为联通图，并把同一“时间戳”的分支点连起来。
    - t_gap: 时间偏移（正值表示在 B 侧向远端方向偏移；负值向近端偏移）
    - ab_bidir: 是否为 A<->B 加双向连接边（默认只加 A->B）
    产物：
      g.ndata['feat']: 节点2D坐标 (A在前、B在后)
      g.edata['elen']: 边长
      g.edata['etype']: 0=A内部边，1=B内部边，2=AB连接边
    """
    # 取坐标到CPU，保证构图一致性
    XA = gA.ndata['feat'].detach().cpu()
    XB = gB.ndata['feat'].detach().cpu()
    NA, NB = XA.size(0), XB.size(0)
    assert NA == k0 + n * k and NB == k0 + n * k, \
        f"节点数不匹配：NA={NA}, NB={NB}, 期望={k0 + n*k}"

    # 基础偏移与节点坐标拼接
    X_all = torch.cat([XA, XB], dim=0)
    offset_B = NA

    # 拿原图边（转CPU int64）
    uA, vA = gA.edges()
    uB, vB = gB.edges()
    uA, vA = uA.detach().cpu().to(torch.int64), vA.detach().cpu().to(torch.int64)
    uB, vB = uB.detach().cpu().to(torch.int64), vB.detach().cpu().to(torch.int64)

    # A内部边 / B内部边（B需要整体偏移）
    src_A, dst_A = uA, vA
    src_B, dst_B = uB + offset_B, vB + offset_B
    # === A/B 的主干与分支索引 ===
    masterA = torch.arange(0, k0, dtype=torch.int64)  # A 主干 [0, k0)
    masterB = offset_B + torch.arange(0, k0, dtype=torch.int64)  # B 主干 [offset_B, offset_B+k0)

    subA_groups = [torch.arange(k0 + j * k, k0 + (j + 1) * k, dtype=torch.int64) for j in range(n)]
    subB_groups = [offset_B + torch.arange(k0 + j * k, k0 + (j + 1) * k, dtype=torch.int64) for j in range(n)]
    ab_src, ab_dst = [], []
    # TODO 应该做最短匹配
    pair_map = list(range(n)) if pair_map is None else pair_map
    for i, j in enumerate(pair_map):
        sAj = subA_groups[i]
        mA = masterA
        sBj = subB_groups[j]
        mB = masterB
        A = torch.cat([mA, sAj], dim=0)
        B = torch.cat([mB, sBj], dim=0)
        LA, LB = A.numel(), B.numel()
        a0 = (k0 - 1) + t_gap
        b0 = (k0 - 1)

        # 计算可遍历的 t 范围，使 A[a0+t] 与 B[b0-t] 都不越界
        # 条件：0 <= a0 + t < LA  且  0 <= b0 - t < LB
        t_lo = max(-a0, b0 - (LB - 1))
        t_hi = min((LA - 1) - a0, b0)

        for t in range(t_lo, t_hi + 1):
            ai = A[a0 + t].to(torch.int64)
            bi = B[b0 - t].to(torch.int64)
            ab_src.append(ai)
            ab_dst.append(bi)
            if ab_bidir:
                ab_src.append(bi)
                ab_dst.append(ai)

    if len(ab_src) == 0:
        ab_src = torch.empty(0, dtype=torch.int64)
        ab_dst = torch.empty(0, dtype=torch.int64)
    else:
        ab_src = torch.tensor(ab_src, dtype=torch.int64)
        ab_dst = torch.tensor(ab_dst, dtype=torch.int64)

    # 拼接所有边
    src = torch.cat([src_A, src_B, ab_src], dim=0)
    dst = torch.cat([dst_A, dst_B, ab_dst], dim=0)

    # 构图
    g = dgl.graph((src, dst), num_nodes=NA + NB, idtype=torch.int64)
    g = g.to(device)
    g.ndata['feat'] = X_all.to(device)

    # 边类型标记：0=A, 1=B, 2=AB
    etype = torch.cat([
        torch.zeros(src_A.numel(), dtype=torch.int64),
        torch.ones(src_B.numel(), dtype=torch.int64),
        torch.full((ab_src.numel(),), 2, dtype=torch.int64)
    ], dim=0).to(device)
    g.edata['etype'] = etype

    # 边长度
    e_src, e_dst = g.edges()
    elen = torch.norm(g.ndata['feat'][e_src] - g.ndata['feat'][e_dst], dim=1, keepdim=True)
    g.edata['elen'] = elen

    return g

# ---- 单个样本 -> DGLGraph ----
def build_one_graph_dgl(k0_seq, branches, directed=True, add_reverse=False, add_edge_len=True, device='cpu'):
    """
    k0_seq:    (k0, 2)   torch.Tensor / np.ndarray
    branches:  (n, k, 2) torch.Tensor / np.ndarray
    directed:  True=有向链, False=无向(会自动补反向边)
    add_reverse: 若 directed=True 且想显式加反向边，可设 True
    add_edge_len: 计算边长度到 g.edata['elen']
    """
    if not isinstance(k0_seq, torch.Tensor): k0_seq = torch.tensor(k0_seq, dtype=torch.float32)
    if not isinstance(branches, torch.Tensor): branches = torch.tensor(branches, dtype=torch.float32)
    k0 = k0_seq.shape[0]
    n, k = branches.shape[0], branches.shape[1]

    # 节点拼接：先主序列，再 n 条分支
    X = torch.cat([k0_seq, branches.reshape(n*k, 2)], dim=0)  # [k0 + n*k, 2]
    N = X.size(0)

    src, dst = [], []

    # 主序列链 0->1->...->k0-1
    if k0 >= 2:
        s = torch.arange(0, k0-1, dtype=torch.int64)
        d = s + 1
        src.append(s); dst.append(d)

    # 分支链 + 主末节点 -> 分支首节点
    main_end = torch.tensor([k0-1], dtype=torch.int64)
    for j in range(n):
        base = k0 + j * k
        # 连接主末 -> 分支首
        src.append(main_end); dst.append(torch.tensor([base], dtype=torch.int64))
        # 分支内部链
        if k >= 2:
            s = torch.arange(base, base + k - 1, dtype=torch.int64)
            d = s + 1
            src.append(s); dst.append(d)

    src = torch.cat(src) if src else torch.empty(0, dtype=torch.int64)
    dst = torch.cat(dst) if dst else torch.empty(0, dtype=torch.int64)

    if not directed:
        # 无向：补反向边
        src = torch.cat([src, dst])
        dst = torch.cat([dst, src[:len(dst)]])  # 注意：上行拼接前的 dst

    if directed and add_reverse:
        # 有向但想显式补反向边
        src = torch.cat([src, dst])
        dst = torch.cat([dst, src[:len(dst)]])

    g = dgl.graph((src, dst), num_nodes=N, idtype=torch.int64).to(device)
    g.ndata['feat'] = X.to(device)  # 节点特征：2 维 (x,y)

    if add_edge_len and g.num_edges() > 0:
        # 计算边长度
        elen = torch.norm(g.ndata['feat'][src.to(device)] - g.ndata['feat'][dst.to(device)], dim=1, keepdim=True)
        g.edata['elen'] = elen

    return g

# ---- 批量 -> DGLBatch ----
def build_batched_graph_dgl(k0_batch, branches_batch, **kwargs):
    """
    k0_batch:       (B, 1, k0, 2)
    branches_batch: (B, n, k, 2)
    kwargs 传给 build_one_graph_dgl（如 directed, add_edge_len, device 等）
    返回：DGL 的 batched 图 (DGLGraph)
    """
    if not isinstance(k0_batch, torch.Tensor): k0_batch = torch.tensor(k0_batch, dtype=torch.float32)
    if not isinstance(branches_batch, torch.Tensor): branches_batch = torch.tensor(branches_batch, dtype=torch.float32)

    B = k0_batch.shape[0]
    graphs = []
    for b in range(B):
        g = build_one_graph_dgl(k0_batch[b, 0], branches_batch[b], **kwargs)
        graphs.append(g)
    return dgl.batch(graphs)

# ====== 用法示例 ======
if __name__ == '__main__':
    B, n, k0, k = 4, 3, 5, 6
    k0_batch = torch.randn(B, 1, k0, 2)
    branches_batch = torch.randn(B, n, k, 2)
    bg = build_batched_graph_dgl(k0_batch, branches_batch, directed=True, add_reverse=False, add_edge_len=True, device='cpu')

    print(bg)                  # DGLGraph(num_nodes=..., num_edges=..., ndata_schemes={'feat': Scheme(shape=(2,), dtype=torch.float32)} ...)
    print(bg.batch_size)       # B
    print(bg.ndata['feat'].shape, bg.edata['elen'].shape)