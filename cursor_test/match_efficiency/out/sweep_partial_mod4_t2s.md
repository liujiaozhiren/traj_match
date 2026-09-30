# Gallery size sweep — 14 methods (scratch, no ckpt)

13 baselines + cursor_mod_4. Online `T_e2e` = `T_query` + `T_gallery`.
Gallery encode/diffusion counted in **online** `T_gallery`.

## T_e2e (s)

| method | 100 | 200 | 500 | 1000 |
|---|---:|---:|---:|---:|
| mod4 | 0.0446 | 0.0799 | 0.3224 | 0.6372 |
| traj2simvec | 0.0581 | 0.1084 | 0.2692 | 0.5305 |

## T_query (s)

| method | 100 | 200 | 500 | 1000 |
|---|---:|---:|---:|---:|
| mod4 | 0.0064 | 0.0063 | 0.0060 | 0.0061 |
| traj2simvec | 0.0027 | 0.0024 | 0.0024 | 0.0024 |

## T_gallery (s)

| method | 100 | 200 | 500 | 1000 |
|---|---:|---:|---:|---:|
| mod4 | 0.0382 | 0.0736 | 0.3164 | 0.6310 |
| traj2simvec | 0.0554 | 0.1060 | 0.2668 | 0.5281 |

## VRAM (MiB)

| method | 100 | 200 | 500 | 1000 |
|---|---:|---:|---:|---:|
| mod4 | 105.6392 | 169.6318 | 358.9995 | 366.5317 |
| traj2simvec | 55.4580 | 56.0298 | 57.7446 | 60.7905 |
