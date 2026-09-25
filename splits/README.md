# Dataset splits

Images stay with their original providers. This folder ships only the **case-ID lists**
used for the reported numbers. One ID per line; IDs are filename stems (no extension).

| Dataset | Files | Counts | How the split is formed |
|---------|-------|--------|-------------------------|
| PH² | `ph2/{train,val,test}.txt` | 80 / 20 / 100 | Full 200-case split. The loader caches `trainx/` in this order, then slices `[0:80]`, `[80:100]`, `[100:200]`. Rebuild the `np/` cache if you change the lists. |
| HAM10000 | `ham10000/{train,val,test}.txt` | 7200 / 1800 / 1015 | Prefix slices of the **sorted** `images/*.jpg` list (10 015 files). |
| ISIC 2017 | `isic2017/{train,val,test}.txt` | 1250 / 150 / 600 | Prefix slices of the **sorted** official **training** JPGs (2000). Challenge val/test folders are not used. |
| ISIC 2018 | `isic2018/{train,val,test}.txt` | 1815 / 259 / 520 | Prefix slices of the **sorted** Task-1 training JPGs (2594). |
| BUSI | `busi/{train,val,test}.txt` | 454 / 64 / 129 | `random.shuffle` of the **sorted** `images/*.png` list with seed **1234**, then 70 / 10 / 20 (`int(0.2 N)` test, `int(0.1 N)` val). |
| Synapse | `synapse/{train,test}.txt` | 2211 slices / 12 volumes | [TransUNet](https://github.com/Beckschen/TransUNet) protocol. The 12 volumes are 3D validation (no held-out test). Used automatically by the loader. |

### Notes

- Skin / BUSI loaders still cache a single `np/X_*.npy` for the **whole** dataset, then apply the index slices above. The lists are the IDs in that cache order so you can audit or rebuild.
- PH² file order on disk is not stable, so `PreparePH2` reads `splits/ph2/` and caches images in that order. Rebuild `np/` if you edit the lists.
- BUSI training uses `args.seed` (launcher default `1234`). The lists assume that seed. A different `--seed` draws a different 70/10/20 split.
- Synapse `train.txt` names slices (`case0031_slice000`); `test.txt` names volumes (`case0008`).
