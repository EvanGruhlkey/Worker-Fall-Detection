# Temporal model runs

Written by `scripts/train_cv.py`. Every number is from grouped cross-validation: each clip is
scored by a model that never saw its scene. Falls caught = alert on the faller within 1.5 s of
impact; false alerts = any other alert, over all held-out footage.

| Run | Settings | Threshold | Falls caught | False alerts | Window AP | Final model |
|---|---|---:|---:|---:|---:|---|
| [2026-11-05_baseline](2026-11-05_baseline/report.md) | gru h64 d0.1, win 30, 2 seeds | 0.99 | 10.5/21 | 6 | 0.636 |  |
| [2026-11-05_cauca-in-folds](2026-11-05_cauca-in-folds/report.md) | gru h64 d0.1, win 30, margin, external in folds, 2 seeds | 0.99 | 12.5/21 | 6.5 | 0.776 |  |
| [2026-11-05_margin](2026-11-05_margin/report.md) | gru h64 d0.1, win 30, margin, 2 seeds | 0.99 | 12/21 | 4.5 | 0.666 |  |
| [2026-11-05_margin-aug3](2026-11-05_margin-aug3/report.md) | gru h64 d0.1, win 30, margin, aug x3, 2 seeds | 0.99 | 12.5/21 | 7.5 | 0.749 |  |
| [2026-11-05_margin-synth200](2026-11-05_margin-synth200/report.md) | gru h64 d0.1, win 30, margin, synth 200, 2 seeds | 0.98 | 10.5/21 | 4 | 0.652 |  |
| [2026-11-05_margin-tcn](2026-11-05_margin-tcn/report.md) | tcn h64 d0.1, win 30, margin, 2 seeds | 0.98 | 13.5/21 | 6.5 | 0.577 |  |
| [2026-11-05_plus-cauca](2026-11-05_plus-cauca/report.md) | gru h64 d0.1, win 30, margin, +caucafall (train), 2 seeds | 0.99 | 13.5/21 | 6 | 0.728 |  |
| [2026-11-05_plus-cauca-aug3](2026-11-05_plus-cauca-aug3/report.md) | gru h64 d0.1, win 30, margin, aug x3, +caucafall (train), 2 seeds | 0.95 | 13.5/21 | 10.5 | 0.755 |  |
| [2026-11-05_rig-only](2026-11-05_rig-only/report.md) | gru h64 d0.1, win 30, margin, 2 seeds | 0.99 | 12/21 | 4.5 | 0.666 |  |
| [2026-11-05_rig-only-win18](2026-11-05_rig-only-win18/report.md) | gru h64 d0.1, win 18, margin, 2 seeds | 0.99 | 10/21 | 2 | 0.654 |  |
