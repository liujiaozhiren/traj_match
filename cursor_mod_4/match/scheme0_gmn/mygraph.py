import torch
import dgl


def build_pair_graph_dgl(gA, gB, k0, n, k, t_gap=0, device="cpu", ab_bidir=True, pair_map=None):
    """
    Migrated from match_model/my/mygraph.py (scheme0 exact behavior).
    """
    XA = gA.ndata["feat"].detach().cpu()
    XB = gB.ndata["feat"].detach().cpu()
    NA, NB = XA.size(0), XB.size(0)
    assert NA == k0 + n * k and NB == k0 + n * k, f"节点数不匹配：NA={NA}, NB={NB}, 期望={k0 + n*k}"

    X_all = torch.cat([XA, XB], dim=0)
    offset_B = NA

    uA, vA = gA.edges()
    uB, vB = gB.edges()
    uA, vA = uA.detach().cpu().to(torch.int64), vA.detach().cpu().to(torch.int64)
    uB, vB = uB.detach().cpu().to(torch.int64), vB.detach().cpu().to(torch.int64)

    src_A, dst_A = uA, vA
    src_B, dst_B = uB + offset_B, vB + offset_B

    masterA = torch.arange(0, k0, dtype=torch.int64)
    masterB = offset_B + torch.arange(0, k0, dtype=torch.int64)
    subA_groups = [torch.arange(k0 + j * k, k0 + (j + 1) * k, dtype=torch.int64) for j in range(n)]
    subB_groups = [offset_B + torch.arange(k0 + j * k, k0 + (j + 1) * k, dtype=torch.int64) for j in range(n)]

    ab_src, ab_dst = [], []
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

    src = torch.cat([src_A, src_B, ab_src], dim=0)
    dst = torch.cat([dst_A, dst_B, ab_dst], dim=0)

    g = dgl.graph((src, dst), num_nodes=NA + NB, idtype=torch.int64)
    g = g.to(device)
    g.ndata["feat"] = X_all.to(device)

    etype = torch.cat(
        [
            torch.zeros(src_A.numel(), dtype=torch.int64),
            torch.ones(src_B.numel(), dtype=torch.int64),
            torch.full((ab_src.numel(),), 2, dtype=torch.int64),
        ],
        dim=0,
    ).to(device)
    g.edata["etype"] = etype

    e_src, e_dst = g.edges()
    elen = torch.norm(g.ndata["feat"][e_src] - g.ndata["feat"][e_dst], dim=1, keepdim=True)
    g.edata["elen"] = elen
    return g


def build_one_graph_dgl(k0_seq, branches, directed=True, add_reverse=False, add_edge_len=True, device="cpu"):
    """
    Migrated from match_model/my/mygraph.py (scheme0 exact behavior).
    """
    if not isinstance(k0_seq, torch.Tensor):
        k0_seq = torch.tensor(k0_seq, dtype=torch.float32)
    if not isinstance(branches, torch.Tensor):
        branches = torch.tensor(branches, dtype=torch.float32)
    k0 = k0_seq.shape[0]
    n, k = branches.shape[0], branches.shape[1]

    X = torch.cat([k0_seq, branches.reshape(n * k, 2)], dim=0)
    N = X.size(0)
    src, dst = [], []

    if k0 >= 2:
        s = torch.arange(0, k0 - 1, dtype=torch.int64)
        d = s + 1
        src.append(s)
        dst.append(d)

    main_end = torch.tensor([k0 - 1], dtype=torch.int64)
    for j in range(n):
        base = k0 + j * k
        src.append(main_end)
        dst.append(torch.tensor([base], dtype=torch.int64))
        if k >= 2:
            s = torch.arange(base, base + k - 1, dtype=torch.int64)
            d = s + 1
            src.append(s)
            dst.append(d)

    src = torch.cat(src) if src else torch.empty(0, dtype=torch.int64)
    dst = torch.cat(dst) if dst else torch.empty(0, dtype=torch.int64)

    if not directed:
        src = torch.cat([src, dst])
        dst = torch.cat([dst, src[: len(dst)]])
    if directed and add_reverse:
        src = torch.cat([src, dst])
        dst = torch.cat([dst, src[: len(dst)]])

    g = dgl.graph((src, dst), num_nodes=N, idtype=torch.int64).to(device)
    g.ndata["feat"] = X.to(device)
    if add_edge_len and g.num_edges() > 0:
        elen = torch.norm(g.ndata["feat"][src.to(device)] - g.ndata["feat"][dst.to(device)], dim=1, keepdim=True)
        g.edata["elen"] = elen
    return g

