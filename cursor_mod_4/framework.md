# cursor_mod_4 方法框架（论文用）

本文档按 **Diffusion / Match / RL** 三个模块整理 `cursor_mod_4` 的实现细节与可直接写入论文的公式。符号与代码默认配置（`cursor_mod_4/cfg.py`）一致。

---

## 0. 问题设定与整体流水线

### 0.1 数据对象

每个匹配样本是一条 **pre–post 轨迹对**（pair），记为

\[
(A, B, L_A, L_B,\ldots),
\]

其中 \(A\) 为 pre 侧观测轨迹，\(B\) 为 post 侧观测轨迹；\(L_A, L_B\) 为对应的标签/完整轨迹（用于 diffusion 监督）。坐标统一投影到局部平面米制 \((x,y)\)（等距近似，原点默认深圳附近）。

关键长度（固定）：

| 符号 | 配置名 | 默认值 | 含义 |
|------|--------|--------|------|
| \(L_0\) | `diff_pre_len` | 8 | 观测头（info / head）点数 |
| \(L\) | `diff_infer_len` | 4 | 待生成尾（tail）点数 |
| \(N\) | `hydra_tail` | 64 | 每个头采样的候选尾条数 |
| \(K\) | `rl_use` / `k_select` | 16 | Match/RL 实际使用的尾条数 |

约定：

- **info / head**：\(\mathbf{h}^{\mathrm{pre}},\mathbf{h}^{\mathrm{post}}\in\mathbb{R}^{L_0\times 2}\)
- **label / GT tail**（仅 diffusion 训练）：\(\mathbf{x}_0^{\mathrm{pre}},\mathbf{x}_0^{\mathrm{post}}\in\mathbb{R}^{L\times 2}\)
- **hydra 候选尾**：\(\mathbf{H}^{\mathrm{pre}},\mathbf{H}^{\mathrm{post}}\in\mathbb{R}^{N\times L\times 2}\)

post 侧在喂入模型前对 \(B\)（及标签）做时间反转 `reversed(B)`，使两端“朝向相遇区域”。

### 0.2 三模块关系

```text
观测头 h^{pre}, h^{post}
        │
        ▼
┌───────────────────┐
│  Diffusion 模块   │  条件生成 N 条尾 → Hydra cache
└─────────┬─────────┘
          │  H^{pre}, H^{post} ∈ R^{N×L×2}
          ▼
┌───────────────────┐
│   Match 模块      │  用头+尾打分，判别是否为真 pair
└─────────┬─────────┘
          │  冻结 Match 作为奖励/教师
          ▼
┌───────────────────┐
│    RL 模块        │  从 N 中选 K 条尾，提升检索指标
└───────────────────┘
```

典型训练顺序：

1. **Diffusion FT**：在 pair 上微调 pre/post 两个 `FixLenDiff84`。
2. **Hydra cache**：对每个 pair 采样 \(N\) 条尾，落盘。
3. **Match FT**：用 cache 的前 \(K\) 条（或选定 \(K\) 条）训练匹配网络。
4. **RL / Offline rank**：冻结 Match，学习从 \(N\) 中选 \(K\) 的选择器。

---

## 1. Diffusion 模块

**代码**：`cursor_mod_4/diffusion/fixlendiff84.py`，`unet84.py`；入口 `entrypoints/ft_diffusion_on_match_pairs.py`；基类 `my_diffusion/base/model.py`。

### 1.1 任务定义

给定条件头 \(\mathbf{c}\in\mathbb{R}^{2\times L_0}\)（通道优先，与实现一致），学习条件分布

\[
p_\theta\!\left(\mathbf{x}_0 \mid \mathbf{c}\right),\qquad \mathbf{x}_0\in\mathbb{R}^{2\times L}.
\]

实现上维护 **两个独立模型**：

- `pre`：\(\mathbf{c}=\mathbf{h}^{\mathrm{pre}}\)，预测 pre 侧尾；
- `post`：\(\mathbf{c}=\mathbf{h}^{\mathrm{post}}\)，预测 post 侧尾。

### 1.2 噪声日程（DDPM）

设总步数 \(T=\) `diffusion_timestamp`（CUDA 默认 200）。线性 \(\beta\) 日程：

\[
\beta_t \;=\; \beta_{\mathrm{start}} + \frac{t}{T-1}\bigl(\beta_{\mathrm{end}}-\beta_{\mathrm{start}}\bigr),\quad
t=0,\ldots,T-1,
\]

默认 \(\beta_{\mathrm{start}}=10^{-4}\)，\(\beta_{\mathrm{end}}=0.05\)。定义

\[
\alpha_t \;=\; 1-\beta_t,\qquad
\bar\alpha_t \;=\; \prod_{s=0}^{t}\alpha_s.
\]

### 1.3 前向加噪

对干净尾 \(\mathbf{x}_0\)，采样 \(t\sim\mathrm{Uniform}\{0,\ldots,T-1\}\)（实现中还做 antithetic：一半 \(t\)、一半 \(T-1-t\)），噪声 \(\boldsymbol{\varepsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})\)：

\[
\mathbf{x}_t \;=\; \sqrt{\bar\alpha_t}\,\mathbf{x}_0 + \sqrt{1-\bar\alpha_t}\,\boldsymbol{\varepsilon}.
\]

### 1.4 噪声预测网络（双分支 UNet84）

网络 \(\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,\mathbf{c},t)\) 为 `AssistTrajUnetModel_8_4`：

- **x 分支**：输入 \(\mathbf{x}_t\in\mathbb{R}^{2\times 4}\)，下采样 \(4\to 2\)；
- **info 分支**：输入 \(\mathbf{c}\in\mathbb{R}^{2\times 8}\)，下采样 \(8\to 4\to 2\)；
- 在长度 2 处 **相加融合** \(h_x + h_{\mathrm{info}}\)，经 mid 块后，用 x 分支 skip 上采样回长度 4，输出 \(\hat{\boldsymbol{\varepsilon}}\in\mathbb{R}^{2\times 4}\)。

时间嵌入：正弦 timestep embedding \(\to\) MLP，注入各 ResNet 块；可选 `extra` 加到 temb。

### 1.5 训练损失

噪声 MSE（主损失）与可选 \(\mathbf{x}_0\) 重建：

\[
\hat{\mathbf{x}}_0 \;=\; \frac{\mathbf{x}_t - \sqrt{1-\bar\alpha_t}\,\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,\mathbf{c},t)}{\sqrt{\bar\alpha_t}},
\]

\[
\mathcal{L}_{\mathrm{diff}}
\;=\;
\underbrace{\bigl\|\boldsymbol{\varepsilon}-\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,\mathbf{c},t)\bigr\|_2^2}_{\mathcal{L}_\varepsilon}
\;+\;
\lambda_{x0}\underbrace{\bigl\|\hat{\mathbf{x}}_0-\mathbf{x}_0\bigr\|_2^2}_{\mathcal{L}_{x0}}.
\]

默认 \(\lambda_{x0}=0\)（`cfg.lambda_x0`）。优化器：AdamW，默认学习率 \(10^{-6}\)。

监督构造（每个 pair）：

\[
\mathbf{c}^{\mathrm{pre}} = \mathrm{xy}(A)[:L_0],\quad
\mathbf{x}_0^{\mathrm{pre}} = \mathrm{xy}(L_A)[-L:],
\]
\[
\mathbf{c}^{\mathrm{post}} = \mathrm{xy}(\mathrm{rev}(B))[:L_0],\quad
\mathbf{x}_0^{\mathrm{post}} = \mathrm{xy}(\mathrm{rev}(L_B))[-L:].
\]

验证指标：对采样 \(\hat{\mathbf{x}}_0\) 与 GT 的米制 RMSE（`rmse_l2_m`）。

### 1.6 反向采样（推理）

从 \(\mathbf{x}_T\sim\mathcal{N}(\mathbf{0},\mathbf{I})\) 出发，\(t=T-1,\ldots,0\)：

\[
\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)
\;=\;
\frac{1}{\sqrt{\alpha_t}}
\left(
\mathbf{x}_t
-
\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}
\boldsymbol{\varepsilon}_\theta(\mathbf{x}_t,\mathbf{c},t)
\right),
\]

\[
\mathbf{x}_{t-1}
\;=\;
\boldsymbol{\mu}_\theta(\mathbf{x}_t,t)
\;+\;
\mathbf{1}_{[t>0]}\sqrt{\beta_t}\,\mathbf{z},\qquad
\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I}).
\]

（实现中方差取 \(\sigma_t=\sqrt{\beta_t}\)，标准简化 DDPM。）

### 1.7 Hydra 多样本采样与 Cache

对同一 \(\mathbf{c}\)，独立采样 \(N\) 次噪声得到 \(N\) 条尾：

\[
\mathbf{H}
\;=\;
\bigl\{\mathbf{x}_0^{(j)}\bigr\}_{j=1}^{N}
\;=\;
\texttt{infer\_from\_noise}(\mathbf{c},\;\texttt{num}=N)
\;\in\;
\mathbb{R}^{N\times 2\times L}
\;\xrightarrow{\text{permute}}\;
\mathbb{R}^{N\times L\times 2}.
\]

`HydraOnlyCache` 存储每个 pair 的

\[
\bigl(\mathbf{H}^{\mathrm{pre}},\mathbf{H}^{\mathrm{post}}\bigr)\in\mathbb{R}^{N_{\mathrm{pairs}}\times N\times L\times 2},
\]

供 Match / RL 复用，避免每次重采样。

---

## 2. Match 模块

**代码**：`cursor_mod_4/match/matcher.py`；训练 `ft_match/train.py`，入口 `entrypoints/run_ft_match.py`。论文中默认只描述 `match_scheme=4`，即基于头点与 Hydra 候选尾 token 的 **Pair Transformer** 匹配器。

### 2.1 任务定义

给定头与 \(K\) 条（或多条）候选尾，输出 **pair 相似度 logit** \(s\in\mathbb{R}\)：

\[
s \;=\; f_\phi\!\left(\mathbf{h}^{\mathrm{pre}},\mathbf{h}^{\mathrm{post}},\mathbf{H}^{\mathrm{pre}}_{1:K},\mathbf{H}^{\mathrm{post}}_{1:K}\right).
\]

训练后在验证集上做 **批内检索**：对 \(k\) 个样本构造分数矩阵 \(\mathbf{S}\in\mathbb{R}^{k\times k}\)，\(\mathbf{S}_{ij}=s(i\to j)\)，正确匹配在对角线上。

### 2.2 输入构造与负样本

默认 Match 微调只用 cache 的前 \(K=\texttt{rl\_use}\) 条尾（非 RL 选择）。

**BCE 模式（`train_loss_mode=bce`）**：batch 内对 post 侧做随机循环移位 derangement \(\pi\)，构造

\[
\begin{aligned}
\text{正样本}&:\;(\mathbf{h}^{\mathrm{pre}}_i,\mathbf{h}^{\mathrm{post}}_i,\mathbf{H}^{\mathrm{pre}}_i,\mathbf{H}^{\mathrm{post}}_i),\; y=1,\\
\text{负样本}&:\;(\mathbf{h}^{\mathrm{pre}}_i,\mathbf{h}^{\mathrm{post}}_{\pi(i)},\mathbf{H}^{\mathrm{pre}}_i,\mathbf{H}^{\mathrm{post}}_{\pi(i)}),\; y=0.
\end{aligned}
\]

**BB-CE 模式（`bb_ce`，推荐用于默认 scheme 4）**：构造全连接 \(B\times B\) logit 矩阵，用对角监督：

\[
\mathcal{L}_{\mathrm{bb}}
\;=\;
\mathrm{CE}\!\left(\mathbf{S},\;\{0,1,\ldots,B-1\}\right)
\;=\;
-\frac{1}{B}\sum_{i=1}^{B}\log\frac{e^{S_{ii}}}{\sum_{j=1}^{B}e^{S_{ij}}}.
\]

### 2.3 Scheme 4：Pair Transformer 匹配器

默认 Match 不显式构造图，也不手工做尾–尾分配，而是把 pre/post 两侧的头点和候选尾点全部组织成 token 序列，让 Transformer 自动学习跨侧关系。

对第 \(i\) 个 pair，输入为

\[
\mathbf{h}^{\mathrm{pre}}_i,\mathbf{h}^{\mathrm{post}}_i\in\mathbb{R}^{L_0\times 2},
\qquad
\mathbf{H}^{\mathrm{pre}}_i,\mathbf{H}^{\mathrm{post}}_i\in\mathbb{R}^{K\times L\times 2}.
\]

先将每侧“头 + 多条尾”展平成点 token：

\[
\mathbf{X}^{\mathrm{pre}}_i
\;=\;
\mathrm{Concat}\!\left(
\mathbf{h}^{\mathrm{pre}}_i,\;
\mathrm{Flatten}_{K,L}(\mathbf{H}^{\mathrm{pre}}_i)
\right)
\in \mathbb{R}^{(L_0+KL)\times 2},
\]

\[
\mathbf{X}^{\mathrm{post}}_i
\;=\;
\mathrm{Concat}\!\left(
\mathbf{h}^{\mathrm{post}}_i,\;
\mathrm{Flatten}_{K,L}(\mathbf{H}^{\mathrm{post}}_i)
\right)
\in \mathbb{R}^{(L_0+KL)\times 2}.
\]

然后将两侧 token 拼接为一个联合序列：

\[
\mathbf{X}_i
\;=\;
\mathrm{Concat}\!\left(\mathbf{X}^{\mathrm{pre}}_i,\mathbf{X}^{\mathrm{post}}_i\right)
\in \mathbb{R}^{2(L_0+KL)\times 2}.
\]

每个二维坐标点通过线性层投影到 \(d\) 维隐藏空间（默认 \(d=128\)）：

\[
\mathbf{E}_i
\;=\;
\mathbf{X}_i\mathbf{W}_{\mathrm{in}}+\mathbf{b}_{\mathrm{in}}
\in \mathbb{R}^{2(L_0+KL)\times d}.
\]

在序列最前加入可学习分类 token：

\[
\mathbf{Z}^{(0)}_i
\;=\;
\mathrm{Concat}\!\left(\mathbf{e}_{\mathrm{cls}},\mathbf{E}_i\right)
\in \mathbb{R}^{(1+2(L_0+KL))\times d}.
\]

Transformer Encoder 通过多头自注意力建模所有 token 之间的关系。第 \(\ell\) 层中，单个注意力头可写为

\[
\mathrm{Attn}(\mathbf{Q},\mathbf{K},\mathbf{V})
\;=\;
\mathrm{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^{\top}}{\sqrt{d_h}}\right)\mathbf{V},
\]

其中

\[
\mathbf{Q}=\mathbf{Z}^{(\ell)}\mathbf{W}_Q,\qquad
\mathbf{K}=\mathbf{Z}^{(\ell)}\mathbf{W}_K,\qquad
\mathbf{V}=\mathbf{Z}^{(\ell)}\mathbf{W}_V.
\]

多头注意力输出后接前馈网络、残差连接与归一化，得到最终序列表示

\[
\mathbf{Z}^{(M)}_i
\;=\;
\mathrm{TransformerEncoder}\!\left(\mathbf{Z}^{(0)}_i\right),
\]

代码中默认 \(M=4\)、注意力头数为 4。取 CLS 位置作为整个 pair 的全局表示：

\[
\mathbf{r}_i
\;=\;
\mathbf{Z}^{(M)}_{i,0}
\in\mathbb{R}^{d}.
\]

最后通过 MLP 输出匹配 logit：

\[
s_i
\;=\;
\mathbf{w}_2^{\top}\,
\mathrm{ReLU}\!\left(\mathbf{W}_1\mathbf{r}_i+\mathbf{b}_1\right)
+b_2.
\]

该设计的直观含义是：pre 侧头、pre 侧多条可能尾、post 侧头、post 侧多条可能尾全部在同一个注意力空间内交互。模型不需要预先决定哪条 pre tail 对应哪条 post tail，而是通过 self-attention 自动学习“哪些候选尾组合更支持该 pair 为真实匹配”。

### 2.4 训练损失（BCE）

\[
\mathcal{L}_{\mathrm{pair}}
\;=\;
\mathrm{BCEWithLogits}(s,y)
\;=\;
-\Bigl[
y\log\sigma(s)+(1-y)\log\bigl(1-\sigma(s)\bigr)
\Bigr].
\]

### 2.5 验证指标

对分数矩阵 \(\mathbf{S}\in\mathbb{R}^{k\times k}\)（真匹配在对角）：

\[
\mathrm{Acc}
\;=\;
\frac{1}{k}\sum_{i=1}^{k}\mathbf{1}\!\left[\arg\max_j S_{ij}=i\right],
\]

\[
\mathrm{HR@}m
\;=\;
\frac{1}{k}\sum_{i=1}^{k}\mathbf{1}\!\left[i\in\mathrm{Top\text{-}}m(S_{i,:})\right],
\]

以及比率 \(\mathrm{R5@10}=\mathrm{HR@5}/\mathrm{HR@10}\) 等；验证 CE 为 \(\mathrm{CE}(\mathbf{S},\mathrm{diag})\)。

---

## 3. RL 模块

**代码**：`cursor_mod_4/rl/linucb_selector.py`，`joint_tail_rank_selector.py`；入口 `entrypoints/run_rl_tail_select.py`，`train_offline_teacher_rank_selector.py`。

目标：在固定头与 \(N\) 条 hydra 候选下，选出 \(K\) 条尾送入 **冻结** Match，使检索更好。两条路线：

1. **在线 bandit**：DeepLinUCB（主路径）；
2. **离线蒸馏**：JointTailRankSelector + ListNet（教师为 Match 的 k=1 arm 分数）。

### 3.1 DeepLinUCB 选择器

#### 3.1.1 候选编码（FiLM）

记 \(\mathbf{c}=[\mathrm{vec}(\mathbf{h}^{\mathrm{pre}});\mathrm{vec}(\mathbf{h}^{\mathrm{post}})]\)，信息编码 \(\mathbf{f}=\mathrm{MLP}(\mathbf{c})\in\mathbb{R}^{H}\)。

对第 \(j\) 条候选尾展平 \(\mathbf{z}_j^{\mathrm{pre}}=\mathrm{MLP}(\mathrm{vec}(\boldsymbol{\tau}_j^{\mathrm{pre}}))\)，FiLM：

\[
\gamma,\beta \;=\; \mathrm{split}\!\bigl(\mathrm{Linear}(\mathbf{f})\bigr),\qquad
\tilde{\mathbf{z}}_j^{\mathrm{pre}}
\;=\;
\gamma^{\mathrm{pre}}\odot\mathbf{z}_j^{\mathrm{pre}}+\beta^{\mathrm{pre}}.
\]

再经 MLP 得表示 \(\mathbf{Z}^{\mathrm{pre}},\mathbf{Z}^{\mathrm{post}}\in\mathbb{R}^{N\times H}\)（默认 \(H=32\)）。

#### 3.1.2 线性 UCB 分数

维护 ridge 统计量（pre/post 各一套），初始化 \(A=\lambda I\)，\(b=\mathbf{0}\)（默认 \(\lambda=1\)）：

\[
\hat{\boldsymbol{\theta}} \;=\; A^{-1}b.
\]

对臂 \(j\)：

\[
\mu_j \;=\; \mathbf{z}_j^\top\hat{\boldsymbol{\theta}},\qquad
\sigma_j \;=\; \sqrt{\mathbf{z}_j^\top A^{-1}\mathbf{z}_j},
\]

\[
\mathrm{UCB}_j \;=\; \mu_j + \alpha\,\sigma_j
\quad(\text{默认 }\alpha=1).
\]

- **独立选择**：对 pre/post 分别 \(\mathrm{Top\text{-}}K(\mathrm{UCB})\)；
- **joint_select**：同一索引集合

\[
\mathrm{UCB}^{\mathrm{joint}}_j
\;=\;
(\mu_j^{\mathrm{pre}}+\mu_j^{\mathrm{post}})
+\alpha\,(\sigma_j^{\mathrm{pre}}+\sigma_j^{\mathrm{post}}).
\]

#### 3.1.3 奖励（相对 baseline 的优势）

Baseline：取候选的前 \(K\) 条（vanilla-\(K\)）。对正样本（\(y=1\)），默认 `bce` 奖励：

对每个选中臂 \(j\)（k=1 喂入 Match）：

\[
\ell_j \;=\; \mathrm{BCEWithLogits}\!\bigl(s(\mathbf{h},\boldsymbol{\tau}_j),\,1\bigr),
\]

\[
r_j^{\mathrm{pos}}
\;=\;
\bar\ell^{\mathrm{base}} - \ell_j^{\mathrm{sel}},
\qquad
\bar\ell^{\mathrm{base}}=\frac{1}{K}\sum_{m=1}^{K}\ell_m^{\mathrm{base}}.
\]

负样本（derangement，\(y=0\)）同理，标签为 0。可选 `bb_ce`：正样本用整批

\[
r^{\mathrm{pos}}
\;=\;
\frac{\mathrm{CE}(\mathbf{S}^{\mathrm{base}})-\mathrm{CE}(\mathbf{S}^{\mathrm{sel}})}{B\cdot K}
\]

（广播到每个选中臂）；负样本仍用 BCE 优势。

奖励批内标准化：

\[
\tilde r \;=\; \mathrm{clip}\!\left(\frac{r-\mathrm{mean}(r)}{\mathrm{std}(r)},\,[-c,c]\right)
\quad(c=3\text{ 默认}).
\]

#### 3.1.4 参数更新

选中臂表示 \(\mathbf{z}_i\)、奖励 \(\tilde r_i\)：

1. **编码器**：\(\min\;\|\mathbf{z}^\top\hat{\boldsymbol{\theta}}-\tilde r\|_2^2\)（pre+post MSE 之和），Adam 更新；
2. **LinUCB 统计**（编码器更新后再算 \(\mathbf{z}'\)）：

\[
A \leftarrow A + \mathbf{z}'{\mathbf{z}'}^\top,\qquad
b \leftarrow b + \tilde r\,\mathbf{z}'.
\]

### 3.2 离线 Joint Tail Ranker

`RichArmScorer` 对每个臂打分 \(u_j\in\mathbb{R}\)：

- 头 token：\(L_0\) pre + \(L_0\) post；
- 尾：Conv1d + Transformer self-attn，再与头做 cross-attn 池化；
- `prepost_sep_fuse`：pre/post 尾分别编码后 concat-MLP 融合再打分。

**ListNet 蒸馏**：教师为冻结 Match 在 k=1 下对 \(N\) 个臂的 logit \(\mathbf{t}\in\mathbb{R}^{N}\)，学生分数 \(\mathbf{u}\)：

\[
\mathcal{L}_{\mathrm{ListNet}}
\;=\;
\mathrm{KL}\!\left(
\mathrm{softmax}(\mathbf{t}/T_t)
\;\big\|\;
\mathrm{softmax}(\mathbf{u}/T_s)
\right).
\]

可选 mini-retrieval 辅助损失（学生 softmax 混合臂分数构造 \(B\times B\) 矩阵，与教师矩阵对齐 + CE）。推理：\(\mathrm{Top\text{-}}K(\mathbf{u})\) 选取同一索引的 pre/post 尾。

### 3.3 评估对比

同一验证协议下对比：

- **match_vanilla**：冻结 Match + 前 \(K\) 条尾；
- **rl_selector / rank_selector**：冻结 Match + 选择器选出的 \(K\) 条尾。

指标同 §2.8（Acc、HR@\(m\)、valid CE）。

---

## 4. 符号速查

| 符号 | 含义 |
|------|------|
| \(L_0,L\) | 头长 8、尾长 4 |
| \(N,K\) | 候选尾数 64、选用数 16 |
| \(\mathbf{x}_t,\bar\alpha_t,\beta_t\) | DDPM 状态与日程 |
| \(\boldsymbol{\varepsilon}_\theta\) | 条件噪声预测 UNet |
| \(s,f_\phi\) | Match logit / 网络 |
| \(\mathbf{S}\) | 批内检索分数矩阵 |
| \(\mathbf{Z},A,b,\alpha\) | LinUCB 表示与统计量 |
| \(r,\tilde r\) | 原始 / 标准化臂奖励 |

---

## 5. 关键文件索引

| 模块 | 核心实现 | 训练/评测入口 |
|------|----------|----------------|
| Diffusion | `diffusion/fixlendiff84.py`, `unet84.py` | `entrypoints/ft_diffusion_on_match_pairs.py` |
| Hydra cache | `ft_match/hydra_cache.py` | （由 match/RL 脚本调用） |
| Match | `match/matcher.py` | `entrypoints/run_ft_match.py` |
| RL (LinUCB) | `rl/linucb_selector.py` | `entrypoints/run_rl_tail_select.py` |
| RL (offline) | `rl/joint_tail_rank_selector.py` | `entrypoints/train_offline_teacher_rank_selector.py` |
| 配置 | `cfg.py` | — |

---

*文档依据仓库 `cursor_mod_4` 当前实现整理，供论文 Method 节直接改写。*
