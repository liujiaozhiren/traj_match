# mod4 超参实验 — 8 项 × 五档

> **8 个主超参**，每个 **5 档**；表中 **★ = 默认对照**（单因子扫描时其余 7 项保持 ★）。  
> 不含 diff/match 的 `lr`·`batch`·`early-stop`，不含 `match_scheme` / `train_loss_mode` 等选项型。

**默认对照（档 3 或标注 ★ 档）**：

```text
T=200  unet_dim=512  num_res_blocks=2  lambda_x0=0
hydra_tail=64  k_select=rl_use=16  match_dim=128  rl_lr=1e-4
```

---

## 总表

| # | 超参 | 管什么 | 改哪里 | 档1 | 档2 | 档3 | 档4 | 档5 | 重跑范围 |
|---|------|--------|--------|-----|-----|-----|-----|-----|----------|
| ① | **`diffusion_timestamp`** | 扩散步数 T；infer ∝ T | `cfg.py` | 25 | 50 | 100 | **200★** | 400 | diff→cache→match→RL |
| ② | **`unet_dim`** | 扩散 UNet 宽 `ch`（勿与 match_dim 混） | `cfg.dim` | 128 | 256 | 384 | **512★** | 768 | 同上；跨档不共享 ckpt |
| ③ | **`num_res_blocks`** | UNet 每 stage ResBlock 数 | `unet84.py` | 1 | 2★ | 3 | 4 | 5 | 同上；decoder 需与 encoder 联动改代码 |
| ④ | **`lambda_x0`** | ε-loss + λ·x₀ 重建 | `--lambda-x0` | **0★** | 0.05 | 0.1 | 0.25 | 0.5 | diff→cache→match→RL |
| ⑤ | **`hydra_tail`** | tail **候选条数** / cache 列数 | `--hydra-tail` | 32 | 48 | **64★** | 96 | 128 | cache→match→RL |
| ⑥ | **`k_select` = `rl_use`** | 从候选里用几条 | `--k-select` / `--rl-use` | 4 | 8 | **16★** | 24 | 32 | 改 k：RL；改 `rl_use`：match+RL |
| ⑦ | **`match_dim`** | scheme4 嵌入维 | `MatchModel(dim=…)` | 64 | 96 | **128★** | 192 | 256 | match→RL |
| ⑧ | **`rl_lr`** | RL selector 学习率 | `run_rl_tail_select.py --lr` | 1e-5 | 3e-5 | **1e-4★** | 3e-4 | 1e-3 | 仅 RL→valid |

**档位说明**

- ①②③④：扩散侧；动则整条上游重来。  
- ⑤⑥：tail 池 vs 选用条数；**恒需 `k_select ≤ hydra_tail`**（本次 tail∈{32…128}、k∈{4…32}，单因子扫互不影响）。  
- ⑦：match 容量；`match_dim % 4 == 0`（默认 `n_heads=4`）。  
- ⑧：最便宜，只重训 RL。

**`lambda_x0` / `num_res_blocks` 默认不在几何中心**：λ 默认 0 放档 1；层数默认 2 放档 2（整数档，无法对称）。

---

## 约束：`hydra_tail` × `k_select`

固定 **hydra_tail=64★** 扫 ⑥ 时，五档 **4, 8, 16, 24, 32** 均可。

固定 **k_select=16★** 扫 ⑤ 时，五档 **32, 48, 64, 96, 128** 均 ≥ 16，⑥ 五档 **4–32** 均可（单因子扫 ⑤ 时 k 锁 16）。

| ⑤ 档 | hydra_tail | 备注 |
|------|------------|------|
| 1 | 32 | k=16 合法 |
| 2 | 48 | |
| 3★ | 64 | P0 cache 档 |
| 4 | 96 | |
| 5 | 128 | 可另增 k=64 探上界（非本次五档） |

单因子扫时：**只动一行，其余保持 ★ 档**。

---

## 各超参简注

### ① `diffusion_timestamp`
DDPM `n_steps`；`infer_from_noise` 循环长度。五档 **{25, 50, 100, 200, 400}**，默认 **200★**（档 4）。

### ② `unet_dim`
`AssistTrajUnet84(ch=…)`。五档固定 **{128, 256, 384, 512, 768}**，默认 **512★**（档 4）。仅 **512** 可直接热启 `pretrain/file/model_para_diff_*.pth`；其余档需 scratch FT 或另训预训。

### ③ `num_res_blocks`
encoder 各 stage 重复块数；**decoder 目前写死 2 层**，扫 ③ 前建议改 `unet84.py` 联动。

### ④ `lambda_x0`
`loss = MSE(ε̂,ε) + λ·MSE(x̂₀,x₀)`。λ=0 为纯 ε 预测（档 1★）。

### ⑤ `hydra_tail`
cache 形状 `(N, hydra_tail, 4, 2)`；不是 `diff_infer_len=4`（点数固定）。五档 **{32, 48, 64, 96, 128}**，默认 **64★**（档 3）。

### ⑥ `k_select` = `rl_use`
RL 选 tail 数；vanilla 用前 k 条；match 读 cache 前 `rl_use` 列。**两参数建议始终相等**。

### ⑦ `match_dim`
`Scheme4PairTransformer` 的 `dim`；`train.py` 现写死 `128`，扫前需暴露 `--match-dim`。  
每侧 token 数 ≈ `8 + k_select×4`（k 变则序列变，已在 ⑥ 体现）。

### ⑧ `rl_lr`
`DeepLinUCBSelector` Adam lr；对数尺度五档。

---

## 单因子扫描规模

| 项 | 五档单扫 runs |
|----|----------------|
| ①–④ 扩散 | 4×5 = 20（各动 diff 链） |
| ⑤ hydra_tail | 5 |
| ⑥ k_select | 5 |
| ⑦ match_dim | 5 |
| ⑧ rl_lr | 5 |
| **合计** | **40**（不全笛卡尔积） |

建议顺序：**⑧ → ⑥ → ⑦ → ⑤ → ④ → ① → ② → ③**（便宜→贵）。

---

## 暂不纳入八项的 match 数值项

| 超参 | 默认 | 若以后要加五档 |
|------|------|----------------|
| `n_layers` | 4 | 2, 3, 4, 5, 6 |
| `n_heads` | 4 | 2, 4, 4, 8, 8（受 match_dim 整除） |
| `out_hidden` | 256 | 128, 192, 256, 384, 512 |

---

## 实验记录模板（一行）

```text
run_id  T  unet_dim  res_blk  lam_x0  tail  k  match_dim  rl_lr  → acc@15%  HR5@15%
base    200  512  2  0  64  16  128  1e-4  ...
```
