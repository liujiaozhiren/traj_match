# Retrieval Case Study (self-contained)

挑一条查询轨迹和它的真值,在同一个固定小候选池上,展示每个方法检索出的 **top5**,
并和真值对比。两个对称方向:

- **方向A `pre_anchor`**:以 PRE 作查询 → 检索 top5 **POST**(真值是对应的 POST)。
- **方向B `post_anchor`**:以 POST 作查询 → 检索 top5 **PRE**(真值是对应的 PRE)。

自动挑选的样例满足:**mod4 的 top1 = 真值**,且其它方法的 top5 尽量覆盖真值。

## 完全自包含

本目录不 import 仓库里的 `cursor_baseline` / `cursor_mod_4`。所有模型代码复制在
[`vendor/`](vendor/) 下(`cursor_baseline`、`cursor_mod_4`、`my_diffusion`、`cursor_mod`、
`match_model`、`model/ST2Vec`);[`_bootstrap.py`](_bootstrap.py) 用 `sys.path` 让这些
副本优先被加载。**训练好的权重(.pth/.pt)是数据产物,按绝对路径从
`cursor_mod_runs/` 与 `cursor_baseline/results/` 读取**,见 [`registry.py`](registry.py)。

两个模型栈各自注册不同的顶层 `cfg`,不能同进程共存,因此 baseline 与 mod4 分别在
**独立子进程**里打分([`worker_baseline.py`](worker_baseline.py) /
[`worker_mod4.py`](worker_mod4.py))。

## 方法集(13 个 + mod4)

- 距离法(无权重):`dtw`、`hausdorff`、`frechet`、`sspd`
- 表征编码器(训练权重):`t3s`、`traj2simvec`、`neutraj`、`trajgat`、`st2vec`、`trajcl`
- 配对模型:`dan`
- 补全模型:`attnmove`(用 `region_vocab` + 补全预测,`pred` 与候选 post 的
  `-mean_step_l2` 作分数)
- 本方法:`mod4`(扩散 hydra 缓存 + MatchModel + RL selector 全流程)

### trajGAT / trajCL 是自己训出来的

原来这俩缺位:`trajgat` 的旧 ckpt(`trajgat_run1`,词表 28642)是在 confile 流水线
(`traj_cmb.pkl`,已不在仓库)上训的,和当前 v2 pkl 的 train 划分构出来的词表(44858)对不上,
没法加载;`trajcl` 仓库里压根没有训练权重。

解决办法:`train_missing_baselines.py` 复用仓库真正的训练循环
(`cursor_baseline.train_repr_baseline.run_repr_baseline`),但把 `_load_pairs` 猴补成
**和 case study 完全一致的 v2 pkl 划分**(`all_pairs_v2_maxdtw.pkl`,ratio 0.1 / seed 42,
train 用升序、与 `worker_baseline` 的 `train_pairs` 同序),这样 trajGAT 的 cell→id 映射
在训练/推理两端一致。产物写到 `cursor_baseline/results/{trajgat,trajcl}_v2pkl_run/`(权重视作数据)。

```bash
python train_missing_baselines.py --encoder trajgat --output-dir \
  ../../cursor_baseline/results/trajgat_v2pkl_run --device cuda:0
python train_missing_baselines.py --encoder trajcl  --output-dir \
  ../../cursor_baseline/results/trajcl_v2pkl_run  --device cuda:1
```

`registry.REPR_CKPT` 用 `_best_ckpt()` 自动挑各 run 目录里 acc 最高的 `best_*.pth`
(当前 `trajgat ep16 acc0.0398`、`trajcl ep6 acc0.0306`)。两者现在都 **strict 加载**。
顺带修了 vendored `trajcl_official._pack_two_views` 的一个 bug:长度为 1 的 cell 序列会让
dual-attention 的 `key_padding_mask` 形状对不上,补成 >=2(与 `_pack_single_view` 一致)。

旧 checkpoint 的兼容加载见 `worker_baseline._tolerant_load`(重命名 `subPart`→`subParts.0`;
对 `neutraj` 的 `lonlat_bounds`、`dan` 的未用 `unmatched_embed` 用 `strict=False`)。

## 运行

```bash
# 复现锁定样例(默认走 --pinned-case out/selected_pairs.json)
conda run --no-capture-output -n traj_match python cursor_test/case_study/compute_selections.py --device cuda:0
conda run --no-capture-output -n traj_match python cursor_test/case_study/plot_case_study.py \
  --data out/case_study.pkl --direction both
```

重新自动挑样例(忽略锁定):`compute_selections.py --no-pin --pool-mode smooth --pool-seed 13`。
`sweep_pool_seed.py` 可扫多个 `--pool-seed`(平滑池),按"mod4 top1 命中 + 其它方法把 GT 排进
top2–5 最多 + 其它方法 top1 命中最少"排序,帮你挑样例。

`compute_selections.py` 会:建固定池 → 跑两个 worker → 合并 `k×k` 分数矩阵 →
两方向各挑 anchor → 落盘 `out/case_study.pkl`。`plot_case_study.py` 出两张图。

仅重画(已有分数):`compute_selections.py --skip-workers`。

## 候选池设计

- 大小由 `--frac` 控制:`frac=0.05` ≈ 验证集 727 的 5% ≈ **36 条**。两方向共用同一池。
- 池 = 在两栈共同的 held-out 验证集(`sample_ratio=0.1`、`sample_seed=42`,两栈
  `random.sample` 同 seed 同结果)里再用 `pool_seed` 子采样。
- 注:小池下的 top1-acc 不等同报告里 `frac=0.15` 的设置,这里是 case study 展示用。

## 锁定样例(已选定,可复现)

记录在 [`out/selected_pairs.json`](out/selected_pairs.json):

| 方向 | 池内位置 | **all_pairs_v2_maxdtw.pkl 中的 pair 下标** | mid | mod4 |
|------|---------|------|-----|------|
| `pre_anchor` | pool#29 | **1582** | `1745913414000` | top1 命中 |
| `post_anchor` | pool#35 | **3466** | `1747293758800` | top1 命中 |

> 这就是你认可的那两对(pre 锚点 = 第 1582 对;post 锚点 = 第 3466 对),来自平滑池
> `frac=0.05, pool_seed=13`。`selected_pairs.json` 里同时存了完整的 36 条 `pool_gidx`、
> 两个锚点的起止经纬度与直度。

`compute_selections.py` 默认 `--pinned-case out/selected_pairs.json`:**存在即复用其精确池子并
强制这两对锚点**,从而稳定重现这张图(GPU 浮点会让个别基线在 rank5 边缘进出 top5,不影响结论)。
要重新自动挑样例:加 `--no-pin`(可配 `--pool-seed` / `--pool-mode`)。

## 输出

- `out/case_study_pre_anchor.png`、`out/case_study_post_anchor.png`:12 格网格(每格 = anchor
  黑实线 + 真值绿虚线 + 该方法 top5,top1 深红其余渐浅并标号 1–5;真值若在 top5 内绿色描边;
  mod4 子图红框高亮)。
- `out/case_study.pkl`:两方向、每方法的 `top5_idx / top5_scores / gt_rank / gt_in_top5`,
  以及池内每条 pre/post 的经纬度坐标。
- `out/scores_baseline.npz`、`out/scores_mod4.npz`:各方法 `k×k` 分数矩阵。
- `out/pool.json`:候选池的全局 pair 下标。
