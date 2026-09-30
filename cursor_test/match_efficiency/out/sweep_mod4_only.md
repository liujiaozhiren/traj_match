# Gallery size sweep — 14 methods (scratch, no ckpt)

13 baselines + cursor_mod_4. Online `T_e2e` = `T_query` + `T_gallery`.
Gallery encode/diffusion counted in **online** `T_gallery`.

## T_e2e (s)

| method | 100 | 200 | 500 | 1000 |
|---|---:|---:|---:|---:|
| mod4 | 0.0448 | 0.0798 | 0.3230 | 0.6371 |

## T_query (s)

| method | 100 | 200 | 500 | 1000 |
|---|---:|---:|---:|---:|
| mod4 | 0.0065 | 0.0061 | 0.0061 | 0.0066 |

## T_gallery (s)

| method | 100 | 200 | 500 | 1000 |
|---|---:|---:|---:|---:|
| mod4 | 0.0383 | 0.0737 | 0.3169 | 0.6305 |

## VRAM (MiB)

| method | 100 | 200 | 500 | 1000 |
|---|---:|---:|---:|---:|
| mod4 | 86.1011 | 150.0938 | 339.4614 | 346.9937 |
