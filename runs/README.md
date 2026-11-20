# Temporal model runs

Written by `scripts/train_cv.py`. Every number is from grouped cross-validation: each clip is
scored by a model that never saw its scene. Falls caught = alert on the faller within 1.5 s of
impact; false alerts = any other alert, over all held-out footage. Honest = threshold chosen
without the fold it is tested on (compare runs on this). On Trial 1 clips = the same 21 falls
Trial 1 was scored on (caught, false alerts). Runs before Trial 2 have neither column.

| Run | Settings | Threshold | Falls caught | False alerts | Honest (caught, false) | On Trial 1 clips | Window AP | Final model |
|---|---|---:|---:|---:|---:|---:|---:|---|
| [2026-11-05_baseline](2026-11-05_baseline/report.md) | gru h64 d0.1, win 30, 2 seeds | 0.99 | 10.5/21 | 6 |  |  | 0.636 |  |
| [2026-11-05_cauca-in-folds](2026-11-05_cauca-in-folds/report.md) | gru h64 d0.1, win 30, margin, external in folds, 2 seeds | 0.99 | 12.5/21 | 6.5 |  |  | 0.776 |  |
| [2026-11-05_margin](2026-11-05_margin/report.md) | gru h64 d0.1, win 30, margin, 2 seeds | 0.99 | 12/21 | 4.5 |  |  | 0.666 |  |
| [2026-11-05_margin-aug3](2026-11-05_margin-aug3/report.md) | gru h64 d0.1, win 30, margin, aug x3, 2 seeds | 0.99 | 12.5/21 | 7.5 |  |  | 0.749 |  |
| [2026-11-05_margin-synth200](2026-11-05_margin-synth200/report.md) | gru h64 d0.1, win 30, margin, synth 200, 2 seeds | 0.98 | 10.5/21 | 4 |  |  | 0.652 |  |
| [2026-11-05_margin-tcn](2026-11-05_margin-tcn/report.md) | tcn h64 d0.1, win 30, margin, 2 seeds | 0.98 | 13.5/21 | 6.5 |  |  | 0.577 |  |
| [2026-11-05_plus-cauca](2026-11-05_plus-cauca/report.md) | gru h64 d0.1, win 30, margin, +caucafall (train), 2 seeds | 0.99 | 13.5/21 | 6 |  |  | 0.728 |  |
| [2026-11-05_plus-cauca-aug3](2026-11-05_plus-cauca-aug3/report.md) | gru h64 d0.1, win 30, margin, aug x3, +caucafall (train), 2 seeds | 0.95 | 13.5/21 | 10.5 |  |  | 0.755 |  |
| [2026-11-05_rig-only](2026-11-05_rig-only/report.md) | gru h64 d0.1, win 30, margin, 2 seeds | 0.99 | 12/21 | 4.5 |  |  | 0.666 |  |
| [2026-11-05_rig-only-win18](2026-11-05_rig-only-win18/report.md) | gru h64 d0.1, win 18, margin, 2 seeds | 0.99 | 10/21 | 2 |  |  | 0.654 |  |
| [2026-11-15_t2-base](2026-11-15_t2-base/report.md) | gru h64 d0.1, win 30, margin, +caucafall+gym (train), 3 seeds | 0.99 | 24/29 | 6.66667 | 24/29, 7.33333 | 19.3333/21, 6.66667 | 0.893 |  |
| [2026-11-15_t2-dedupe](2026-11-15_t2-dedupe/report.md) | gru h64 d0.1, win 30, margin, dedupe, +caucafall+gym (train), 3 seeds | 0.99 | 24/29 | 5.66667 | 24/29, 7 | 19.6667/21, 5.66667 | 0.854 |  |
| [2026-11-15_t2-drop015](2026-11-15_t2-drop015/report.md) | gru h64 d0.1, win 30, margin, +caucafall+gym (train), drop>=0.15, 3 seeds | 0.99 | 24/29 | 5.66667 | 24/29, 6.66667 | 19.3333/21, 5.66667 | 0.893 |  |
| [2026-11-15_t2-gap12](2026-11-15_t2-gap12/report.md) | gru h64 d0.1, win 30, margin, gap 12, +caucafall+gym (train), 3 seeds | 0.99 | 24.6667/29 | 7.66667 | 24.6667/29, 8.33333 | 19.6667/21, 7.66667 | 0.849 |  |
| [2026-11-15_t2-low-x3](2026-11-15_t2-low-x3/report.md) | gru h64 d0.1, win 30, margin, low x3, +caucafall+gym (train), 3 seeds | 0.99 | 23.6667/29 | 6 | 23.6667/29, 7 | 19.3333/21, 6 | 0.860 |  |
| [2026-11-15_t2-mirror](2026-11-15_t2-mirror/report.md) | gru h64 d0.1, win 30, margin, mirror, +caucafall+gym (train), 3 seeds | 0.99 | 25/29 | 5.66667 | 25.3333/29, 7.33333 | 20.3333/21, 5.66667 | 0.906 |  |
| [2026-11-15_t2-no-gym](2026-11-15_t2-no-gym/report.md) | gru h64 d0.1, win 30, margin, no gym, +caucafall (train), 3 seeds | 0.99 | 23.3333/29 | 8 | 23.3333/29, 8.66667 | 19.3333/21, 8 | 0.865 |  |
| [2026-11-15_t2-smooth01](2026-11-15_t2-smooth01/report.md) | gru h64 d0.1, win 30, margin, smooth 0.1, +caucafall+gym (train), 3 seeds | 0.99 | 19/29 | 2.66667 | 19.6667/29, 3 | 15.3333/21, 2.66667 | 0.863 |  |
| [2026-11-15_t2-win18](2026-11-15_t2-win18/report.md) | gru h64 d0.1, win 18, margin, +caucafall+gym (train), 3 seeds | 0.99 | 22/29 | 8 | 22/29, 9 | 17.6667/21, 8 | 0.767 |  |
| [2026-11-15_t2-with-slow](2026-11-15_t2-with-slow/report.md) | gru h64 d0.1, win 30, margin, +test-only in training, +caucafall+gym (train), 3 seeds | 0.99 | 23.3333/29 | 6 | 23.6667/29, 7 | 19.3333/21, 5.66667 | 0.867 |  |
| [2026-11-16_t2-best](2026-11-16_t2-best/report.md) | gru h64 d0.1, win 30, margin, mirror, dedupe, +caucafall+gym (train), drop>=0.15, 3 seeds | 0.99 | 25/29 | 4.33333 | 25/29, 4.66667 | 20.3333/21, 4.33333 | 0.879 | yes |
| [2026-11-19_t3-base](2026-11-19_t3-base/report.md) | gru h64 d0.1, win 30, margin, mirror, dedupe, +caucafall+gym (train), drop>=0.15, 3 seeds | 0.99 | 34.3333/42 | 9 | 34.3333/42, 10.3333 | 20/21, 2 | 0.796 |  |
| [2026-11-19_t3-low-x2](2026-11-19_t3-low-x2/report.md) | gru h64 d0.1, win 30, margin, mirror, dedupe, low x2, +caucafall+gym (train), drop>=0.15, 3 seeds | 0.99 | 32.6667/42 | 9 | 34/42, 9.66667 | 19/21, 1.33333 | 0.800 |  |
| [2026-11-19_t3-low-x3](2026-11-19_t3-low-x3/report.md) | gru h64 d0.1, win 30, margin, mirror, dedupe, low x3, +caucafall+gym (train), drop>=0.15, 3 seeds | 0.99 | 33.6667/42 | 8.33333 | 33.6667/42, 9 | 19.6667/21, 2.66667 | 0.784 |  |
| [2026-11-19_t3-smooth005](2026-11-19_t3-smooth005/report.md) | gru h64 d0.1, win 30, margin, mirror, dedupe, smooth 0.05, +caucafall+gym (train), drop>=0.15, 3 seeds | 0.99 | 34.6667/42 | 10 | 34.6667/42, 10 | 20.3333/21, 1.66667 | 0.795 |  |
| [2026-11-19_t3-smooth01](2026-11-19_t3-smooth01/report.md) | gru h64 d0.1, win 30, margin, mirror, dedupe, smooth 0.1, +caucafall+gym (train), drop>=0.15, 3 seeds | 0.99 | 32.3333/42 | 9.66667 | 32/42, 11 | 18.3333/21, 1.66667 | 0.739 |  |
| [2026-11-19_t3-thr](2026-11-19_t3-thr/report.md) | gru h64 d0.1, win 30, margin, mirror, dedupe, +caucafall+gym (train), drop>=0.15, 3 seeds | 0.999 | 31/42 | 7 | 31/42, 7.33333 | 17.6667/21, 1.33333 | 0.801 |  |
