# Match 效率对比 — 14 方法 × 1 query vs N gallery（E2E 检索）

> **任务定义**：库里已有 **N 条候选轨迹**（CLI 参数 `--gallery-size`，**默认 4000**）。来一条 **未见过的 query**，在 N 条里找 top-1。  
> query = **pre 侧**，gallery = **post 侧**。

效率实验量 **一次完整检索** 的 E2E 时延与峰值内存。  
**实现约定**：`T_gallery` 一律按 **批量并行** 测（见 §1.3），禁止用 for-loop 串行 N 次作为报告值。

---

## 0. 方法清单（14 + 1 子模式）

| # | 方法 | 离线索引 | 在线 `T_gallery` 实现形态 |
|---|------|----------|---------------------------|
| 1–4 | DTW / Hausdorff / Fréchet / SSPD | N 条轨迹 | **1 次** batched 距离矩阵 `(1,N)` |
| 5–10 | 表征双塔 | N×128 embed | **1 次** `cdist(1,N)` |
| 11 | DAN | N×128 post 塔 | **1 次** batched MLP on `(N,256)` |
| 12 | TAT | 同 DAN 或 N×post 塔 | **1 次** batched graph_update+MLP（若开 graph） |
| 13 | AttnMove | N 条 post 轨迹 | **1 次** broadcast L2 `(1,T,2)` vs `(N,T,2)` |
| 14 | cursor_mod_4 | N 份 info+hydra(post) | **1 次** batched Match `(N,…)`；query hydra **在线 diffusion** |
| 14b | mod4 + RL | 同上 | **1 次** batched selector+match |

---

## 1. 三类核心指标 + 全局参数

### 1.0 CLI 参数（所有脚本统一）

| 参数 | 默认 | 含义 |
|------|------|------|
| `--gallery-size` | `4000` | gallery 规模 **N** |
| `--gallery-batch` | `4000`（或显存不够时调小） | `T_gallery` **单 kernel 内并行宽度**；`N > gallery-batch` 时按 chunk 并行，**禁止**逐条 for-loop |
| `--hydra-tail` | `64` | diffusion 采样 tail 数（selector 再从 64 里选 16） |
| `--k-select` | `16` | mod4 vanilla 用前 k 条 tail；RL 由 selector 选 k |
| `--device` | `cuda:0` | |

报告主表必须带 `N` 和实际 `gallery-batch`（若 chunk 了也写明）。

### 1.1 三类指标

| 指标 | 符号 | 定义 |
|------|------|------|
| **参数量** | `P` | 模型 + 辅助可学习表，`numel` 计数，dim 统一 **128**（§4） |
| **内存 / 显存** | `M` | **一次完整检索**（1 query vs N）的进程 **峰值** |
| **耗时** | `T` | **一次完整检索**墙钟时间 |

另列：`S_index`（索引存储）、`T_index`（离线建库时间）。

### 1.2 耗时分解

```
T_e2e = T_query + T_gallery(N) + T_argmax
```

| 项 | 含义 |
|----|------|
| `T_query` | 仅 query：预处理、encode、补全、**query 侧在线 diffusion**（mod4） |
| `T_gallery(N)` | 对 **全部 N 个** gallery 打分——**逻辑上 N 路，实现上 batched 并行** |
| `T_argmax` | top-1（通常忽略） |

**P 与 N 无关**；**M / T 与 N 强相关**。

### 1.3 批量并行约定（重要）

你说得对：**N 路打分之间没有数据依赖，应并行，不应串行 for-loop。**

| 类型 | 逻辑 | 标准实现（报告用这个） |
|------|------|------------------------|
| 距离法 | N 个距离 | `pairwise_neg_distance_matrix(pre=(1,L,2), post=(N,L,2))` → `(1,N)`；过大则 `gallery-batch` chunk，**每 chunk 仍并行** |
| 表征 | N 个 L2 | `cdist(z_q.unsqueeze(0), Z_gallery)` 一次 |
| DAN / TAT | N 个 MLP | `u_q` broadcast：`cat([u_q.expand(N,-1), U_gallery], -1)` → **一次** `pair_mlp` → `(N,)` |
| AttnMove | N 个轨迹 L2 | `pred (1,T,2)` vs `B (N,T,2)` broadcast → `(N,)` |
| mod4 | N 个 pair match | `info_q/hydra_q` repeat → `MatchModel` **一次** `B=N`（或 `gallery-batch` chunk） |
| mod4+RL | N 个 selector+match | `choose` + `match` 均按 `B=N` batch（与 `eval_match_rl_noise_blend_grid` 同形） |

**Chunk 规则**：仅当 `N * per_pair_activation > 显存` 时，设 `gallery-batch < N`：

```
T_gallery = sum_{c in chunks} T_batch_parallel(c)    # 通常 1 次；chunk 时 2~few 次
```

**禁止**：`for j in range(N): score_j = f(q, g_j)` 作为 `T_gallery` 报告值（可作 debug 对照，不进主表）。

`M` 峰值取 **最宽那个 batch**（往往 `gallery-batch == N` 时最大）。

### 1.4 dim=128（仅用于 P）

`embed_dim = rnn_dim = match_dim = 128`；T3S `lstm_hidden=128`；TrajGAT `d_model=128`。

---

## 2. 各方法 E2E Pipeline（参数 N，`--gallery-size`）

约定：gallery `{g_j}_{j=1}^N` **事先已知**；query `q` **事先不在库里**。

---

### 2.1 几何距离（DTW / Hausdorff / Fréchet / SSPD）

```
离线:  S_index = N × (12×2)
在线:  T_query  = preprocess(q)
       T_gallery = pairwise_dist_batched(q, G) → (1,N)   # 并行，非 for-loop
       T_e2e     = T_query + T_gallery + T_argmax
```

实现：`pairwise_neg_distance_matrix`，`pre=(1,12,2)`，`post=(N,12,2)`；`chunk_queries` 与 `gallery-batch` 对齐。

---

### 2.2 表征双塔（Traj2SimVec / T3S / NeuTraj）

```
离线:  Z ∈ R^{N×128}
在线:  T_query  = preprocess(q) + encode_pre(q)
       T_gallery = cdist(z_q, Z)              # 一次 GEMM，N 路并行
```

---

### 2.3 TrajGAT / 2.4 TrajCL / ST2Vec

同 §2.2 的 batched scan；query 侧离散化/构图进 `T_query`，gallery 侧进离线 `T_index`。

---

### 2.5 Deep Affinity Network（DAN）

```
离线:  U ∈ R^{N×128}  (post 塔)
在线:  T_query  = preprocess(q) + backbone_pre(q) → u_q  (1,128)
       T_gallery = pair_mlp( cat([u_q.expand(N,-1), U], dim=-1) )  → (N,)   # 一次 forward
```

**语义**：逻辑上 N 个 `MLP([u_q; u_j])`；**实现**上矩阵 batch，GPU 同步并行。

---

### 2.6 Tracklet Association Tracker（TAT）

**`use_graph_update=False`**：同 DAN，一次 batched MLP。

**`use_graph_update=True`**：

```
T_gallery = assoc_mlp( graph_update( stack([h_q.expand(N,-1), U], dim=1) ) )   # (N,2,C) batch
```

2-node self-attn 在 dim=1 上 batch 成 `(N,2,128)`，**仍可一次并行**，不是 N 次 Python 循环。

---

### 2.7 AttnMove（AM）

```
离线:  gallery post 轨迹 B ∈ R^{N×T×2}
在线:  T_query  = 补全 1 次 → pred (1,T,2)
       T_gallery = -mean_t L2( pred[:,None], B )  → (N,)   # broadcast，一次
```

---

### 2.8 cursor_mod_4（scheme4）— 在线 diffusion + 与 N 路并行 match

**部署假设**（按你的要求）：

- **Gallery 已知**：`info_post_j`、`hydra_post_j` 在 **离线建索引** 时用 diffusion 生成（与现 `build_hydra_only_cache` 相同，但对 gallery 条目做）。
- **Query 未知**：上线时 **在线**跑 `pre_diff.infer_from_noise(head_pre_q, num=hydra_tail)` 得到 `hydra_pre_q`。

```
离线 (T_index):
  对 j=1..N:
    info_post_j  ← pair 的 8 点 xy
    hydra_post_j ← post_diff.infer_from_noise(head_post_j, num=64)   # gallery 侧 diffusion
  S_index = N × (info + hydra_post)

在线:
  T_query  = raw(q) → info_pre_q
           + pre_diff.infer_from_noise(head_pre_q, num=64) → hydra_pre_q   # 仅 query，与 N 无关
           + 取 vanilla 前 k_select 条 tail（或 RL 前保留全 64 给 selector）

  T_gallery = 一次 batched pair match（与 N 个 gallery **同步推理**）:
    info_pre  = info_pre_q.expand(N, -1, -1)           # (N, 8, 2)
    info_post = info_post_gallery                        # (N, 8, 2)
    hydra_pre = hydra_pre_q_selected.expand(N, ...)      # (N, k_sel, 4, 2)
    hydra_post= hydra_post_gallery_selected              # (N, k_sel, 4, 2)
    scores    = MatchModel(...)                          # (N,)  — 单 kernel

  T_e2e = T_query + T_gallery + T_argmax
```

代码路径对齐：`eval_match_rl_noise_blend_grid` 里 `B1=1, B2=N` 的 repeat/broadcast；效率实验设 `B1=1, gallery-batch=N`。

| 项 | 说明 |
|----|------|
| `T_query` | **必含** query 侧 diffusion（`FixLenDiff84.infer_from_noise`，`num=hydra_tail`） |
| `T_gallery` | **逻辑** N 个 pair；**实现** 1× batched Transformer，`B=N` |
| `M` 峰值 | 通常在 `T_gallery`：`O(N × tokens × dim)` 激活；N=4000 可能放不进显存 → 用 `--gallery-batch` chunk |
| Gallery hydra | **离线**生成，**不算**在线 `T_query` |

#### cursor_mod_4 + RL selector

```
T_gallery = batched selector.choose(info_q, info_post, hydra_pre_q^{64}, hydra_post^{64})
            → batched MatchModel
```

`choose` 支持 `B=N`（`linucb_selector` 对 batch 维循环的是 **UCB top-k**，计算图仍在 GPU 上 batch 编码）；与 eval 脚本一致，**整批 N 路并行**，不按 j 串行。

**不再提供 cache 查表列**——query 一律在线 diffusion。

---

## 3. E2E 对比总表

| 方法 | `T_query`（在线） | `T_gallery`（batched 并行） | 离线 `T_index` 含 diffusion？ |
|------|-------------------|----------------------------|-------------------------------|
| DTW 等 | preprocess | `dist(1,N)` 一次 | 否 |
| NeuTraj 等 | encode q | `cdist(1,N)` | 否 |
| DAN / TAT | pre 塔 | `MLP(N,)` 一次 | 否 |
| AttnMove | 补全 1 次 | `L2(1,N)` broadcast | 否 |
| **cursor_mod_4** | **pre diffusion×1** | **`Match(N,)` 一次** | **是（gallery post hydra）** |
| mod4+RL | 同左 | `selector+match(N,)` | 是 |

**数量级直觉**（N 相同、均 batched；mod4 的 `T_query` 因 diffusion 抬高）：

```
T_gallery:  mod4_RL ≥ mod4 ≫ DTW ≫ AttnMove ≫ DAN ≈ NeuTraj
T_query:    mod4 ≫ 其余（多一次 diffusion）
T_e2e:      需实测（mod4 query 重 + gallery 重 的权衡）
```

---

## 4. 参数量 P

（不变）`P = P_model@128 + P_aux`；gallery 索引与 hydra 浮点 → `S_index`。

---

## 5. 实验怎么跑

### 5.1 数据

- `--gallery-size N`：从 `pairs_pkl` 取 N 条作 gallery（默认 4000）。
- query：held-out pair 的 **pre 侧**，确保不在 gallery 的 post 索引逻辑里混淆即可。

### 5.2 测量

```text
1. 离线: build_gallery_index(N) → T_index, S_index
         mod4: gallery post 跑 post_diff.infer_from_noise
2. 在线 warmup 1 次（含 query diffusion + batched gallery）
3. reset peak memory
4. time T_query（preprocess + encode/补全 + query diffusion）
5. time T_gallery（单次 batched；若 chunk 则 sum chunks，仍无 per-item loop）
6. T_e2e = T_query + T_gallery + T_argmax；M = peak
7. 50 queries → median；记录 N, gallery-batch
```

### 5.3 主表列

| method | P | S_index | N | gallery-batch | M_e2e | T_e2e | T_query | T_gallery |

---

## 6. 实现顺序

1. `config.py` — `--gallery-size`, `--gallery-batch`, `--hydra-tail`, …
2. `build_gallery_index.py` — 含 mod4 gallery post diffusion
3. `run_retrieval_e2e.py` — 统一 batched `T_gallery`；mod4 query 走 `pre_diff.infer_from_noise`
4. 先打通：**DTW、NeuTraj、DAN、mod4（online diffusion + batched N）**
