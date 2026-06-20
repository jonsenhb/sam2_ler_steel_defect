# NEU Conv Ablation Summary

| method | best_miou_pct | best_epoch | delta_vs_std | patches | inclusion | scratches | early_stop | params_M |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Std-LoRA | 77.08 | 2 | — | 69.56 | 82.14 | 79.55 | Y | 0.72 |
| Conv-LoRA | 76.17 | 4 | -0.92% | 70.21 | 80.75 | 77.54 | Y | 0.73 |
| MAP-LoRA (A) | 76.85 | 2 | -0.23% | 69.50 | 83.43 | 77.60 | Y | 0.73 |
| Phase-Adaptive (B) | 76.32 | 8 | -0.76% | 69.47 | 81.40 | 78.10 | Y | 0.73 |
| Rank-Adaptive (C) | 76.61 | 4 | -0.47% | 70.55 | 79.69 | 79.59 | Y | 0.82 |
