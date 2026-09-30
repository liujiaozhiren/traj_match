# cursor_mod_4 RL 模块详解（论文用）

本文档专门介绍 **论文最终采用的 RL 尾选择方法**：`DeepLinUCBSelector` + `run_rl_tail_select.py`。  
与 `framework.md` 的全局框架文档互补；此处聚焦 RL 的问题定义、算法细节、训练/评测协议与实现约定。

---

## 0. 采用结论（可直接写进论文）

| 项目 | 最终采用 |
|------|----------|
| 方法路线 | **DeepLinUCB**（在线 contextual bandit） |
| 备选未入主结果 | `JointTailRankSelector + ListNet`（`train_offline_teacher_rank_selector.py`） |
| 选择方式 | **pre/post 独立 Top-K**（`joint_select=False`） |
| 正样本训练奖励 | **`bb_ce`**：完整选中 tail set 的 batch retrieval CE 优势 |
| 负样本训练奖励 | **per-arm k=1 BCE** 优势 |
| 推理/验证 | 将完整 Top-K tail set 送入冻结 Match（set-level） |

**代码入口**：`entrypoints/run_rl_tail_select.py`  
**核心实现**：`rl/linucb_selector.py`

---

## 1. 问题定义

### 1.1 背景

Diffusion 模块对每个轨迹 pair 的观测头 \(\mathbf{h}^{\mathrm{pre}},\mathbf{h}^{\mathrm{post}}\) 采样 \(N\) 条候选尾（Hydra）：

\[
\mathbf{H}^{\mathrm{pre}},\mathbf{H}^{\mathrm{post}}
\in\mathbb{R}^{N\times L\times 2},
\qquad N=64,\; L=4.
\]

Match 模块（scheme-4 Pair Transformer，训练后**冻结**）在输入「头 + 多条尾」时输出 pair 相似度 logit。若直接使用前 \(K=16\) 条尾（vanilla-\(K\)），检索效果有限。

**RL 的目标**：从 \(N\) 条候选中选出 \(K\) 条尾，使冻结 Match 在 k×k 批内检索上优于 vanilla-\(K\)。

### 1.2 形式化

对每个样本 \(i\)，观测为

\[
\mathcal{O}_i
=
\bigl(
\mathbf{h}^{\mathrm{pre}}_i,
\mathbf{h}^{\mathrm{post}}_i,
\mathbf{H}^{\mathrm{pre}}_i,
\mathbf{H}^{\mathrm{post}}_i
\bigr).
\]

选择器 \(\pi\) 输出两侧索引：

\[
\mathcal{I}^{\mathrm{pre}}_i,\mathcal{I}^{\mathrm{post}}_i
\subset\{1,\ldots,N\},
\quad |\mathcal{I}|=K.
\]

选中尾集合：

\[
\hat{\mathbf{H}}^{\mathrm{pre}}_i
=
\{\mathbf{H}^{\mathrm{pre}}_{i,j}\}_{j\in\mathcal{I}^{\mathrm{pre}}_i},
\qquad
\hat{\mathbf{H}}^{\mathrm{post}}_i
=
\{\mathbf{H}^{\mathrm{post}}_{i,j}\}_{j\in\mathcal{I}^{\mathrm{post}}_i}.
\]

送入冻结 Match 得分数 \(s_i=f_{\mathrm{match}}(\mathbf{h}^{\mathrm{pre}}_i,\mathbf{h}^{\mathrm{post}}_i,\hat{\mathbf{H}}^{\mathrm{pre}}_i,\hat{\mathbf{H}}^{\mathrm{post}}_i)\)。

训练时以 Match 反馈构造奖励，更新选择器；Match 权重始终冻结。

---

## 2. 方法总览：Deep Contextual LinUCB

整体为 **深度特征 + 线性 UCB** 的两阶段结构：

```text
观测 (h^pre, h^post, H^pre, H^post)
        │
        ▼
┌─────────────────────────┐
│ CandidateEncoder (FiLM) │  → 每条臂嵌入 z_j^pre, z_j^post ∈ R^H
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ LinUCB (pre / post 各一套) │  → UCB_j = μ_j + α·σ_j
└───────────┬─────────────┘
            │
            ▼
   独立 Top-K 选臂 → 送入冻结 Match → 奖励 → 更新 encoder + (A,b)
```

默认超参（P0 / baseline）：

| 参数 | 值 |
|------|-----|
| \(N\) (`hydra_tail`) | 64 |
| \(K\) (`k_select`) | 16 |
| 表示维 \(H\) (`rep_dim`) | 32 |
| UCB 探索系数 \(\alpha\) | 1.0（部分 ckpt 为 0.05） |
| Ridge \(\lambda\) | 1.0 |
| 学习率 | \(10^{-4}\) |
| `joint_select` | **False** |
| `train_reward_mode` | **bb_ce**（P0 规范） |

---

## 3. 候选编码：Pair-conditioned FiLM

### 3.1 输入

- 头：\(\mathbf{h}^{\mathrm{pre}},\mathbf{h}^{\mathrm{post}}\in\mathbb{R}^{L_0\times 2}\)，\(L_0=8\)
- 候选尾：\(\mathbf{H}^{\mathrm{pre}},\mathbf{H}^{\mathrm{post}}\in\mathbb{R}^{N\times L\times 2}\)

### 3.2 共享 pair 上下文

\[
\mathbf{f}
=
\mathrm{MLP}\!\left(
\mathrm{vec}(\mathbf{h}^{\mathrm{pre}})
\;\Vert\;
\mathrm{vec}(\mathbf{h}^{\mathrm{post}})
\right)
\in\mathbb{R}^{H}.
\]

### 3.3 单臂尾编码（第 \(j\) 条）

\[
\mathbf{u}_j^{\mathrm{pre}}
=
\mathrm{MLP}\!\left(\mathrm{vec}(\mathbf{H}^{\mathrm{pre}}_j)\right),
\qquad
\mathbf{u}_j^{\mathrm{post}}
=
\mathrm{MLP}\!\left(\mathrm{vec}(\mathbf{H}^{\mathrm{post}}_j)\right).
\]

FiLM 调制（\(\gamma,\beta\) 由 \(\mathbf{f}\) 产生，pre/post 各一套）：

\[
\tilde{\mathbf{z}}_j^{\mathrm{pre}}
=
\gamma^{\mathrm{pre}}\odot\mathbf{u}_j^{\mathrm{pre}}+\beta^{\mathrm{pre}},
\qquad
\tilde{\mathbf{z}}_j^{\mathrm{post}}
=
\gamma^{\mathrm{post}}\odot\mathbf{u}_j^{\mathrm{post}}+\beta^{\mathrm{post}}.
\]

再经 MLP 得最终臂表示：

\[
\mathbf{z}_j^{\mathrm{pre}},\mathbf{z}_j^{\mathrm{post}}
\in\mathbb{R}^{H}.
\]

### 3.4 重要性质：评分是 per-arm，不是 set-level

- 臂 \(j\) 的嵌入只依赖 \((\mathbf{h}^{\mathrm{pre}},\mathbf{h}^{\mathrm{post}},\mathbf{H}^{\mathrm{pre}}_j,\mathbf{H}^{\mathrm{post}}_j)\)。
- **不会**在打分 arm \(j\) 时读取其他候选尾 \(\mathbf{H}_{j'}\) 或整个候选集。
- pre/post 两侧各维护一套 UCB，可产生**不同的 Top-K 索引**（见 §4）。

---

## 4. 选臂：独立 pre/post LinUCB Top-K

### 4.1 线性回报模型

pre 侧维护 \((A^{\mathrm{pre}},b^{\mathrm{pre}})\)，post 侧维护 \((A^{\mathrm{post}},b^{\mathrm{post}})\)，初始化：

\[
A=\lambda I,\qquad b=\mathbf{0},\qquad \lambda=1.
\]

参数估计：

\[
\hat{\boldsymbol{\theta}}=A^{-1}b.
\]

### 4.2 UCB 分数

对臂 \(j\)：

\[
\mu_j=\mathbf{z}_j^{\top}\hat{\boldsymbol{\theta}},
\qquad
\sigma_j=\sqrt{\mathbf{z}_j^{\top}A^{-1}\mathbf{z}_j},
\qquad
\mathrm{UCB}_j=\mu_j+\alpha\sigma_j.
\]

### 4.3 独立 Top-K（论文采用）

\[
\mathcal{I}^{\mathrm{pre}}=\mathrm{Top\text{-}K}(\mathrm{UCB}^{\mathrm{pre}}),
\qquad
\mathcal{I}^{\mathrm{post}}=\mathrm{Top\text{-}K}(\mathrm{UCB}^{\mathrm{post}}).
\]

**不强制** \(\mathcal{I}^{\mathrm{pre}}=\mathcal{I}^{\mathrm{post}}\)。  
代码默认 `joint_select=False`；已保存 ckpt 中 `joint_select=False`。

> 备选（未采用）：`joint_select=True` 时用联合分数  
> \(\mathrm{UCB}^{\mathrm{joint}}_j=(\mu^{\mathrm{pre}}_j+\mu^{\mathrm{post}}_j)+\alpha(\sigma^{\mathrm{pre}}_j+\sigma^{\mathrm{post}}_j)\)，  
> 两侧共享同一索引集合。

---

## 5. 奖励设计

奖励由**冻结 Match** 给出，采用「相对 vanilla-\(K\) baseline 的优势」。

### 5.1 Baseline

对每个 batch，baseline 取候选池前 \(K\) 条：

\[
\mathbf{H}^{\mathrm{pre}}_{\mathrm{base}}=\mathbf{H}^{\mathrm{pre}}_{1:K},
\qquad
\mathbf{H}^{\mathrm{post}}_{\mathrm{base}}=\mathbf{H}^{\mathrm{post}}_{1:K}.
\]

### 5.2 正样本（\(y=1\)）：`bb_ce` set-level 奖励（P0 采用）

构造 batch 内 \(B\times B\) 检索 logit 矩阵 \(\mathbf{S}\)，对角为正样本。  
对 baseline 与 selector 选中集合分别计算 batch CE：

\[
\mathcal{L}_{\mathrm{base}}
=
\mathrm{CE}\!\left(\mathbf{S}^{\mathrm{base}},\{0,\ldots,B-1\}\right),
\qquad
\mathcal{L}_{\mathrm{sel}}
=
\mathrm{CE}\!\left(\mathbf{S}^{\mathrm{sel}},\{0,\ldots,B-1\}\right).
\]

优势：

\[
\Delta=\mathcal{L}_{\mathrm{base}}-\mathcal{L}_{\mathrm{sel}}.
\]

将标量优势均分到本 batch 每个选中臂（共 \(B\cdot K\) 个）：

\[
r^{\mathrm{pos}}_m
=
\frac{\Delta}{B\cdot K},
\qquad m=1,\ldots,BK.
\]

这是 **set-level 正样本奖励**：\(\mathcal{L}_{\mathrm{sel}}\) 用的是完整 Top-K pre/post tail set 一起送入 Match 后的检索损失，而非单臂 k=1。

### 5.3 负样本（\(y=0\)）：per-arm k=1 BCE 优势

对 post 侧做 derangement \(\pi\) 构造负 pair。  
对每条选中臂 \(j\)，仅用该臂单条尾（k=1）计算 BCE：

\[
\ell^{\mathrm{base}}_j
=
\mathrm{BCE}\!\left(
f_{\mathrm{match}}(\mathbf{h}^{\mathrm{pre}},\mathbf{h}^{\mathrm{post}}_{\pi},\boldsymbol{\tau}^{\mathrm{base}}_j),\,0
\right),
\]

\[
\ell^{\mathrm{sel}}_j
=
\mathrm{BCE}\!\left(
f_{\mathrm{match}}(\mathbf{h}^{\mathrm{pre}},\mathbf{h}^{\mathrm{post}}_{\pi},\boldsymbol{\tau}^{\mathrm{sel}}_j),\,0
\right).
\]

\[
r^{\mathrm{neg}}_j
=
\bar\ell^{\mathrm{base}}-\ell^{\mathrm{sel}}_j,
\qquad
\bar\ell^{\mathrm{base}}=\frac{1}{K}\sum_{m=1}^{K}\ell^{\mathrm{base}}_m.
\]

负样本始终为 **per-arm k=1** 奖励。

### 5.4 奖励归一化

批内 z-score + clip（默认开启）：

\[
\tilde r
=
\mathrm{clip}\!\left(
\frac{r-\mathrm{mean}(r)}{\mathrm{std}(r)},\,-c,\,c
\right),
\qquad c=3.
\]

### 5.5 备选奖励模式 `bce`（代码默认，P0 未采用）

若 `train_reward_mode=bce`，正/负样本均用 §5.3 的 per-arm k=1 BCE 优势。  
P0 规范与重训脚本（`run_rl_retrain_v2cache_shenzhen.sh`、`run_tier.py`）使用 **`bb_ce`**。

---

## 6. 参数更新

每个训练 step（正样本一次 `choose` + `update`，负样本再一次）：

### 6.1 深度编码器

用当前 \(\hat{\boldsymbol{\theta}}\) 对选中臂做回归：

\[
\mathcal{L}_{\mathrm{enc}}
=
\left\|
\mathbf{Z}^{\mathrm{pre}}_{\mathrm{sel}}\hat{\boldsymbol{\theta}}^{\mathrm{pre}}-\tilde{\mathbf{r}}
\right\|_2^2
+
\left\|
\mathbf{Z}^{\mathrm{post}}_{\mathrm{sel}}\hat{\boldsymbol{\theta}}^{\mathrm{post}}-\tilde{\mathbf{r}}
\right\|_2^2.
\]

Adam 更新 encoder。

### 6.2 LinUCB 充分统计

encoder 更新后，用新表示 \(\mathbf{z}'\) 做 ridge 更新：

\[
A \leftarrow A + \mathbf{z}'{\mathbf{z}'}^{\top},
\qquad
b \leftarrow b + \tilde r\,\mathbf{z}'.
\]

pre/post 两侧独立更新各自的 \((A,b)\)。

---

## 7. 训练流程

### 7.1 数据与依赖

1. **Pairs**：7-tuple 列表 `(mid, A, B, L_A, L_B, F_A, F_B)`
2. **Hydra cache**：`HydraOnlyCache`，形状 `(N_{\mathrm{pairs}}, 64, 4, 2)` × pre/post
3. **冻结 Match ckpt**：scheme-4，`match_scheme=4`

pair 顺序、train/valid 划分须与 cache 构建时一致（`sample_ratio=0.1`, `sample_seed=42`）。

### 7.2 单个 epoch

对每个 train batch：

1. 取头 \(\mathbf{h}^{\mathrm{pre}},\mathbf{h}^{\mathrm{post}}\) 与 64 条候选尾
2. 构造负样本：post 侧 derangement
3. **正样本**：`choose` → 算 `bb_ce` 奖励 → `update_batch`
4. **负样本**：`choose` → 算 k=1 BCE 奖励 → `update_batch`
5. valid：k×k retrieval，对比 selector vs vanilla16

### 7.3 Early stopping

监控 valid retrieval Acc；连续 `early_stop=15` 个 epoch 无提升则停止。  
保存 `best_selector_k16_acc*.pt`。

### 7.4 Checkpoint 内容

```text
SelectorCheckpoint:
  meta: {k_select, rep_dim, alpha, lambda_, joint_select, diff_pre_len, diff_infer_len}
  encoder_state: CandidateEncoder 权重
  A_pre, b_pre, A_post, b_post: LinUCB 统计量
```

---

## 8. 推理与评测

### 8.1 推理（set-level）

验证与最终评测时，**始终**将完整 Top-K 尾集合送入冻结 Match：

\[
s
=
f_{\mathrm{match}}\!\left(
\mathbf{h}^{\mathrm{pre}},
\mathbf{h}^{\mathrm{post}},
\hat{\mathbf{H}}^{\mathrm{pre}}_{1:K},
\hat{\mathbf{H}}^{\mathrm{post}}_{1:K}
\right).
\]

这与训练时 `bb_ce` 正样本奖励、valid CE 的计算方式一致。

### 8.2 k×k 批内检索

对 valid 子集大小 \(k\)，构造分数矩阵 \(\mathbf{S}\in\mathbb{R}^{k\times k}\)：

\[
S_{ij}
=
f_{\mathrm{match}}(\mathbf{h}^{\mathrm{pre}}_i,\mathbf{h}^{\mathrm{post}}_j,\hat{\mathbf{H}}^{\mathrm{pre}}_i,\hat{\mathbf{H}}^{\mathrm{post}}_j).
\]

真匹配在对角线 \(S_{ii}\)。

### 8.3 指标

\[
\mathrm{Acc}
=
\frac{1}{k}\sum_{i=1}^{k}\mathbf{1}\!\left[\arg\max_j S_{ij}=i\right],
\]

\[
\mathrm{HR@}m
=
\frac{1}{k}\sum_{i=1}^{k}\mathbf{1}\!\left[i\in\mathrm{Top\text{-}}m(S_{i,:})\right],
\]

\[
\mathcal{L}_{\mathrm{valid}}
=
\mathrm{CE}(\mathbf{S},\{0,\ldots,k-1\}).
\]

同时报告 vanilla16（取前 16 条尾）作为对照。

### 8.4 主评测入口

`entrypoints/eval_match_rl_noise_blend_grid.py`：

- `match_vanilla`：冻结 Match + 前 K 条尾
- `rl_selector`：冻结 Match + `DeepLinUCBSelector` 选出的 K 条尾

---

## 9. 三层机制对照（论文写作要点）

| 阶段 | 粒度 | 说明 |
|------|------|------|
| **UCB 选臂打分** | per-arm | 每条候选尾独立嵌入与 UCB；不读全候选集 |
| **正样本训练奖励** | set-level | `bb_ce`：完整 Top-K set 的 batch retrieval CE 优势 |
| **负样本训练奖励** | per-arm (k=1) | 单条尾 BCE 优势 |
| **验证 / 推理** | set-level | 完整 Top-K set 送入 Match |

论文中应明确区分：**选臂是 per-arm bandit，但 Match 反馈与最终评测是 set-level**。

---

## 10. 与备选方案的关系

仓库中存在另一条未入主结果的路线：

| | DeepLinUCB（采用） | JointTailRank + ListNet（备选） |
|--|-------------------|--------------------------------|
| 脚本 | `run_rl_tail_select.py` | `train_offline_teacher_rank_selector.py` |
| 学习范式 | 在线 bandit + Match 奖励 | 离线蒸馏 + ListNet |
| ckpt 命名 | `best_selector_k*.pt` | `best_rank_k*.pt` |
| 主评测 | `eval_match_rl_noise_blend_grid.py` | `eval_tail_select_valid_fracs.py` |

论文 Method 只写 DeepLinUCB 即可；ListNet ranker 可作为附录或未来工作。

---

## 11. 典型运行命令

深圳 P0 重训（与 baseline 规范一致）：

```bash
conda run --no-capture-output -n traj_match python \
  cursor_mod_4/entrypoints/run_rl_tail_select.py \
  --output-dir cursor_data_proc/out/rl_retrain_v2cache_s4 \
  --pairs-pkl cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl \
  --cache cursor_mod_runs/match_ft_84/hydra_only_cache_84.pt \
  --match-ckpt cursor_mod_runs/match_ft_84/best_match_s4_ep41_acc0.1590.pth \
  --match-scheme 4 \
  --hydra-tail 64 \
  --k-select 16 \
  --train-reward-mode bb_ce \
  --fixed-valid-seed 42 \
  --device cuda:0
```

成都：

```bash
bash cursor_mod_run_chengdu/scripts/run_rl_tail_chengdu.sh
```

---

## 12. 已登记成品路径（参考）

| 数据集 | Selector ckpt |
|--------|---------------|
| 深圳 baseline | `cursor_mod_runs/rl_tail_select_s4_84/best_selector_k16_acc0.1498_ep20.pt` |
| 成都 | `cursor_mod_run_chengdu/rl_tail_scheme4_cuda/best_selector_k16_acc0.0578_ep7.pt` |

成都 ckpt meta 含 `joint_select: false`。

---

## 13. 符号速查

| 符号 | 含义 |
|------|------|
| \(N,K\) | 候选尾数 64、选用数 16 |
| \(\mathbf{z}_j\) | 第 \(j\) 条臂的 deep 表示 |
| \(\mathrm{UCB}_j\) | LinUCB 探索-利用分数 |
| \(\mathcal{I}^{\mathrm{pre}},\mathcal{I}^{\mathrm{post}}\) | 独立选出的索引集合 |
| \(\Delta\) | `bb_ce` 正样本 batch CE 优势 |
| \(\tilde r\) | 归一化后的臂奖励 |
| vanilla-\(K\) | 直接取候选池前 \(K\) 条尾 |

---

*依据 `cursor_mod_4` 当前实现与 P0/baseline 运行规范整理，供论文 RL / Tail Selection 小节直接改写。*
