# cursor_test — 效率实验（自包含，不依赖 cursor_baseline / cursor_mod_4）

**1 query vs N gallery** 检索效率；学习型方法统一 **dim=128**。

## 运行

```bash
# 小规模 smoke（N=32）
bash cursor_test/match_efficiency/run_smoke.sh

# 完整 N=4000（需 GPU 显存足够或调小 --gallery-batch）
conda run --no-capture-output -n traj_match python cursor_test/match_efficiency/run_e2e.py \
  --methods dtw,neutraj,dan,mod4,mod4_rl \
  --gallery-size 4000 \
  --gallery-batch 512 \
  --device cuda:0

# 仅参数量
conda run --no-capture-output -n traj_match python cursor_test/match_efficiency/run_e2e.py \
  --count-params-only --methods dan,mod4
```

## 目录

| 文件 | 作用 |
|------|------|
| `constants.py` | `EMBED_DIM=128` 等 |
| `models/` | 14 方法自包含模型 |
| `gallery.py` | 离线建库（mod4 gallery 侧 diffusion） |
| `retrieval.py` | E2E：`T_query + T_gallery`（gallery 批量并行） |
| `run_e2e.py` | CLI：`--gallery-size`, `--gallery-batch` |
| `match_efficiency/PLAN.md` | 设计文档 |

