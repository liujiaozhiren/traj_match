# 超参实验 — 复用规划（先理清楚，再写代码）

> 数据**只用**深圳 v2：`cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl`（7279 行）。  
> 逻辑与 `cursor_mod_4` 一致；不读 traj_cmb、成都、gap pkl。  
> 单因子扫描：**只动 1 个超参的 5 档，其余 7 项保持 ★**。

---

## 1. ★ 默认对照（P0 / baseline_star）

```text
T=200  unet_dim=512  num_res_blocks=2  lambda_x0=0
hydra_tail=64  k_select=rl_use=16  match_dim=128  rl_lr=1e-4
```

### 1.1 全局固定（不在 8 项里扫，所有实验相同）

| 项 | 值 | 说明 |
|----|-----|------|
| pairs | `cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl` | 唯一 pairs 源 |
| `match_scheme` | 4 | scheme-4 pair transformer |
| `diff_pre_len` / `diff_infer_len` | 8 / 4 | 几何长度，非超参 |
| `sample_ratio` / `sample_seed` | 0.1 / 42 | train 子采样 |
| match `lr` / `train_batch` / `early_stop` | 1e-5 / 32 / 10 | `run_ft_match.py` 默认 |
| match `valid_sample_frac` / `valid_n_passes` | 0.15 / 3 | match 训练内 valid |
| `train_loss_mode` | **bb_ce** | 新训 match/RL 统一用；见 §1.3 |
| RL `train_reward_mode` | bb_ce | `run_rl_tail_select.py` |
| RL `max_epochs` / `early_stop` | 200 / 15 | gap retrain 脚本同款 |
| RL `fixed_valid_seed` | 42 | |
| valid 评测 | α=1.0；fracs=15/10/5/2% | `eval_match_rl_noise_blend_grid.py` |

扩散 FT 侧（非 8 项）：`cfg.diff_lr`、`cfg.batch` 等跟 mod4 `cfg.py`，与 `diff_ft_pairs131_meters_84` 当时一致即可。

### 1.2 ★ 已有成品路径（P0 可直接读，不必重训）

| 阶段 | 文件 | 校验 |
|------|------|------|
| pairs | `cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl` | 7279 行 |
| diff pre | `cursor_mod_runs/diff_ft_pairs131_meters_84/best_pre_ep289_rmse5.908m.pth` | unet_dim=512 |
| diff post | `cursor_mod_runs/diff_ft_pairs131_meters_84/best_post_ep242_rmse4.678m.pth` | 同上 |
| diff 热启（仅 dim=512） | `cursor_mod_4/pretrain/file/model_para_diff_pre131.17892456.pth` + `..._post...` | 新训 diff 时 init |
| hydra cache | `cursor_mod_runs/match_ft_84/hydra_only_cache_84.pt` | `(7279, 64, 4, 2)`；meta `hydra_tail=64` |
| match | `cursor_mod_runs/match_ft_84/best_match_s4_ep41_acc0.1590.pth` | scheme4；`MatchModel(dim=128)` |
| RL selector | `cursor_mod_runs/rl_tail_select_s4_84/best_selector_k16_acc0.1498_ep20.pt` | meta `k_select=16`；默认 `lr=1e-4` |

**不要用**：`cursor_mod_run_chengdu/**`（10000 行 cache，与 7279 不对齐）；gap 专用 RL（`cursor_data_proc/out/gap*_rl_retrain_*`）不是全局 ★ baseline。

### 1.3 已知不一致（接受或一次性对齐）

- `match_ft_84` 当时很可能用默认 `train_loss_mode=bce` 训的；后续 gap/RL 脚本用 `bb_ce`。**最大化复用** → P0 match **frozen 直接用 ep41**；只有重训 match 的 sweep 才统一 `bb_ce`。
- `rl_tail_select_s4_84` 与 ep41 match 是否严格同一次 pipeline 无 meta 记录；ep20 selector 与 ep41 match 是近期 gap 实验默认组合，作为 ★ RL 占位合理。若要做严格 P0 复现，可另跑一轮 `rl_lr=1e-4` 固化到 `hyper-parameter/out/baseline_star/rl/`。

### 1.4 未来代码输出目录（规划）

```text
cursor_test/hyper-parameter/
  registry/baseline_star.json    # 指向 §1.2 各路径
  out/
    baseline_star/               # P0 valid 结果；可选固化 RL
    sweep_08_rl_lr/tier_{1..5}/
    sweep_06_k_select/tier_{1..5}/
    ...
```

`tier_k` 若与 ★ 完全一致 → **skip train，registry 指回 §1.2**。

---

## 2. 依赖链（什么变了必须重跑什么）

```text
diff FT (pre+post)
    → hydra_only_cache
        → match FT
            → RL selector
                → valid (grid.json)
```

| 变动的超参 | 必须重跑 |
|------------|----------|
| ① T、② unet_dim、③ num_res_blocks、④ lambda_x0 | diff → cache → match → RL → valid |
| ⑤ hydra_tail | cache → match → RL → valid |
| ⑥ k_select（= rl_use） | match → RL → valid（cache 列够用时**可读同一 cache 文件**，只取前 k 列） |
| ⑦ match_dim | match → RL → valid |
| ⑧ rl_lr | **仅** RL → valid |

---

## 3. 八个单因子实验 — 每档「其余超参」+ 复用 / 重训

下面「固定」= 该 sweep 内除「本行变动档」外全部为 ★。  
「★档可直接读」= 该 tier 与 P0 一致，指向 §1.2，不必重训。

---

### ⑧ `rl_lr`（最先跑，最便宜）

| | 值 |
|--|-----|
| **变动** | `rl_lr` ∈ {1e-5, 3e-5, **1e-4★**, 3e-4, 1e-3} |
| **固定** | ①–⑦ 全部 ★ |

| Tier | rl_lr | 直接读（reuse） | 必须训 |
|------|-------|-----------------|--------|
| 1 | 1e-5 | pairs, diff×2, cache, match | RL |
| 2 | 3e-5 | 同上 | RL |
| 3 ★ | 1e-4 | **全部 §1.2（含 RL ep20）** | 仅 valid（可选跳过 RL） |
| 4 | 3e-4 | pairs, diff×2, cache, match | RL |
| 5 | 1e-3 | 同上 | RL |

**5 个 RL run 共享**：`hydra_only_cache_84.pt` + `best_match_s4_ep41_acc0.1590.pth`。

---

### ⑥ `k_select` = `rl_use`

| | 值 |
|--|-----|
| **变动** | k ∈ {4, 8, **16★**, 24, 32} |
| **固定** | ①–⑤⑦⑧ 全部 ★；`hydra_tail=64` |

| Tier | k | 直接读 | 必须训 |
|------|---|--------|--------|
| 1 | 4 | pairs, diff×2, **cache（取 [:4]）** | match, RL |
| 2 | 8 | 同上 cache [:8] | match, RL |
| 3 ★ | 16 | **§1.2 全套** | 仅 valid |
| 4 | 24 | pairs, diff×2, cache [:24] | match, RL |
| 5 | 32 | 同上 cache [:32] | match, RL |

约束：`k ≤ hydra_tail(64)`，五档均合法。

---

### ⑦ `match_dim`

| | 值 |
|--|-----|
| **变动** | dim ∈ {64, 96, **128★**, 192, 256}（需 `% 4 == 0`） |
| **固定** | ①–⑥⑧ 全部 ★ |

| Tier | match_dim | 直接读 | 必须训 |
|------|-----------|--------|--------|
| 1 | 64 | pairs, diff×2, cache | match, RL |
| 2 | 96 | 同上 | match, RL |
| 3 ★ | 128 | **§1.2 全套** | 仅 valid |
| 4 | 192 | pairs, diff×2, cache | match, RL |
| 5 | 256 | 同上 | match, RL |

match ckpt **不能跨 dim 共享**；RL selector 与 frozen match 绑定，每档都要新 RL。

---

### ⑤ `hydra_tail`

| | 值 |
|--|-----|
| **变动** | tail ∈ {32, 48, **64★**, 96, 128} |
| **固定** | ①–④⑥⑦⑧ 全部 ★；**k=16** |

| Tier | hydra_tail | 直接读 | 必须训 |
|------|------------|--------|--------|
| 1 | 32 | pairs, diff×2 | cache, match, RL |
| 2 | 48 | 同上 | cache, match, RL |
| 3 ★ | 64 | **§1.2 全套** | 仅 valid |
| 4 | 96 | pairs, diff×2 | cache, match, RL |
| 5 | 128 | 同上 | cache, match, RL |

约束：`k_select=16 ≤ tail` 对五档均成立（最小 tail=32）。

---

### ④ `lambda_x0`

| | 值 |
|--|-----|
| **变动** | λ ∈ {**0★**, 0.05, 0.1, 0.25, 0.5} |
| **固定** | ①②③⑤⑥⑦⑧ 全部 ★ |

| Tier | lambda_x0 | 直接读 | 必须训 |
|------|-----------|--------|--------|
| 1 ★ | 0 | **§1.2 全套** | 仅 valid |
| 2 | 0.05 | pairs；dim=512 可用 pretrain init | diff, cache, match, RL |
| 3 | 0.1 | 同上 | 全链 |
| 4 | 0.25 | 同上 | 全链 |
| 5 | 0.5 | 同上 | 全链 |

---

### ① `diffusion_timestamp` (T)

| | 值 |
|--|-----|
| **变动** | T ∈ {25, 50, 100, **200★**, 400} |
| **固定** | ②–⑧ 全部 ★ |

| Tier | T | 直接读 | 必须训 |
|------|---|--------|--------|
| 1 | 25 | pairs | 全链 |
| 2 | 50 | pairs | 全链 |
| 3 | 100 | pairs | 全链 |
| 4 ★ | 200 | **§1.2 全套** | 仅 valid |
| 5 | 400 | pairs | 全链 |

改 T 需改 `cfg.py`（或 hyper 代码注入）；infer 步数变 → cache 必重算。

---

### ② `unet_dim` (`cfg.dim`)

| | 值 |
|--|-----|
| **变动** | ch ∈ {128, 256, 384, **512★**, 768} |
| **固定** | ①③–⑧ 全部 ★ |

| Tier | unet_dim | 直接读 | 必须训 |
|------|----------|--------|--------|
| 1 | 128 | pairs | 全链（**无** pretrain 热启） |
| 2 | 256 | pairs | 全链 |
| 3 | 384 | pairs | 全链 |
| 4 ★ | 512 | **§1.2 全套** + `pretrain/file/model_para_diff_*.pth` 作 init | tier4 可读成品；新训 diff 时热启 |
| 5 | 768 | pairs | 全链 |

**跨档不共享** diff / cache / match / RL ckpt。

---

### ③ `num_res_blocks`

| | 值 |
|--|-----|
| **变动** | blocks ∈ {1, **2★**, 3, 4, 5} |
| **固定** | ①②④–⑧ 全部 ★ |

| Tier | num_res_blocks | 直接读 | 必须训 |
|------|----------------|--------|--------|
| 1 | 1 | pairs；512 可尝试 pretrain 部分加载 | 全链（结构不匹配可能 strict=False） |
| 2 ★ | 2 | **§1.2 全套**（当前 unet 即 2） | 仅 valid |
| 3–5 | 3/4/5 | pairs | 全链；**需先改 `unet84.py` decoder 联动** |

---

## 4. 汇总表：40 runs 里多少能 skip

| Sweep | 可 skip 训练的 tier | 需新训 tier 数 | 新训阶段 |
|-------|---------------------|----------------|----------|
| ⑧ rl_lr | tier 3（★） | 4 | RL×4 |
| ⑥ k_select | tier 3（★） | 4 | match+RL×4 |
| ⑦ match_dim | tier 3（★） | 4 | match+RL×4 |
| ⑤ hydra_tail | tier 3（★） | 4 | cache+match+RL×4 |
| ④ lambda_x0 | tier 1（★） | 4 | 全链×4 |
| ① T | tier 4（★） | 4 | 全链×4 |
| ② unet_dim | tier 4（★） | 4 | 全链×4 |
| ③ num_res_blocks | tier 2（★） | 4 | 全链×4 |

**若 P0 valid 已记录**：8 个 ★ tier 共 **8 次 valid 可合并为 1 次**（同一套 ckpt）。  
**实际新训量**：约 **32 组训练任务** + **40 次 valid**（或 32 valid 若 8 个 ★ tier 只 valid 一次）。

---

## 5. 写代码时的复用策略（预告）

1. **`registry/baseline_star.json`**：固化 §1.2 路径 + meta（N=7279, tail=64, k=16, …）。
2. **每个 sweep runner**：根据 `sweep_id` + `tier` 查表 §3 → 决定 `reuse` 列表 vs `train` 列表。
3. **pairs**：永远 `--pairs-pkl cursor_mod_4/pipeline/all_pairs_v2_maxdtw.pkl`。
4. **cache**：`k ≤ tail` 时同一 `hydra_only_cache_84.pt` 只读前 k 列，不必为每个 k 存 5 份 cache（仅 sweep ⑤ 要按 tail 重新生成）。
5. **★ tier 检测**：hyper 配置 hash 与 baseline_star 一致 → `skip_train=True`，只跑 valid 写 `grid.json`。
6. **从 mod4 抄逻辑**：`run_ft_match.py` / `run_rl_tail_select.py` / `eval_match_rl_noise_blend_grid.py` 薄封装进 `hyper-parameter/`，加 cfg 注入（T, dim, res_blocks, match_dim, lambda_x0）。

建议执行顺序：**⑧ → ⑥ → ⑦ → ⑤ → ④ → ① → ② → ③**（与 README 一致）。
