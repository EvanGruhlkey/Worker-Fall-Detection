# Temporal model run: t2-dedupe

2026-11-15T22:32:04 · commit `3fa0f41+changes` · manifest.json: 161 clips, 101 falls

## Headline

At the chosen gate threshold **0.99** (mean of 3 seeds), held out by 5-fold grouped cross-validation (every clip tested once, by a model that never saw its scene):

- **Falls caught: 24 of 29** (83%), alert within 1.5 s of impact
- **False alerts: 5.66667** in 6.8 min of non-fall footage (5 on the normal-work clips)
- Median alert time: +0.08 s relative to impact
- Late alerts (faller, after the catch window): 1
- Window-level: average precision 0.85, precision 0.82 / recall 0.84 at this threshold
- **Honest estimate (threshold chosen without the tested fold): 24 of 29 caught, 7 false alerts.** Each fold's threshold (0.85, 0.96, 0.97, 0.99) was picked from the other folds only, so these are the numbers to compare between runs.
- Seed to seed: falls caught 24 to 24, false alerts 3 to 8 (the run-to-run noise; differences between runs smaller than this are not real)

> One false alert in 6.8 min is 9/hour, so this data cannot show the 1/hour target directly; the threshold is chosen for the fewest false alerts seen, then the most falls caught.

## Data and settings

- 39 clips (29 with a fall), 36 scene groups, 11.9998 fps, windows of 30 frames (2.5 s), stride 1
- Model: GRU, hidden 64, dropout 0.1, up to 40 epochs (best by validation), learning rate 0.002
- Ambiguous windows (0-50% of a fall) left out: yes
- Augmentation: 0 augmented copies per training clip; synthetic clips added: 0
- Label smoothing: 0; low-posture normal windows weighted x1
- Tracking: a lost track continues for up to 5 frames (0.42 s); duplicate boxes on one body removed
- Gate: debounce 5 frames (0.42 s), cooldown 30 s
- Fast-drop check: off
- Seeds: 0, 1, 2
- Tested clips by source: rig 39 (29 falls)
- External clips added to every training set (never tested): caucafall 88 (38 falls), gym 22 (22 falls)

## Threshold trade-off

| Threshold | Falls caught | False alerts | Per hour | Normal-clip false | Median alert time | Window precision | Window recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 25/29 | 20.3333 | 179 | 14.3333 | -0.06 s | 0.63 | 0.94 |
| 0.10 | 25.6667/29 | 17.6667 | 156 | 13.6667 | -0.04 s | 0.67 | 0.93 |
| 0.15 | 26/29 | 16.6667 | 147 | 13 | -0.03 s | 0.68 | 0.93 |
| 0.20 | 25.6667/29 | 15.6667 | 138 | 12 | -0.03 s | 0.70 | 0.92 |
| 0.25 | 25.6667/29 | 14.6667 | 129 | 11 | -0.03 s | 0.71 | 0.92 |
| 0.30 | 25.6667/29 | 14 | 124 | 11 | +0.01 s | 0.71 | 0.92 |
| 0.35 | 25.6667/29 | 14 | 124 | 11 | +0.01 s | 0.72 | 0.91 |
| 0.40 | 25.6667/29 | 13 | 115 | 10 | +0.01 s | 0.72 | 0.91 |
| 0.45 | 25.6667/29 | 12.6667 | 112 | 9.66667 | +0.01 s | 0.73 | 0.91 |
| 0.50 | 25.3333/29 | 12.6667 | 112 | 9.66667 | +0.01 s | 0.73 | 0.91 |
| 0.55 | 25.6667/29 | 12.3333 | 109 | 9.66667 | +0.01 s | 0.74 | 0.91 |
| 0.60 | 25.6667/29 | 12.3333 | 109 | 9.66667 | +0.01 s | 0.74 | 0.90 |
| 0.65 | 25.6667/29 | 11.6667 | 103 | 9.33333 | +0.01 s | 0.75 | 0.90 |
| 0.70 | 25.3333/29 | 11.3333 | 100 | 9 | +0.01 s | 0.76 | 0.90 |
| 0.75 | 25.3333/29 | 10.6667 | 94 | 9 | +0.01 s | 0.76 | 0.89 |
| 0.80 | 25.3333/29 | 10.6667 | 94 | 9 | +0.01 s | 0.77 | 0.89 |
| 0.85 | 25.3333/29 | 10.6667 | 94 | 9 | +0.01 s | 0.78 | 0.89 |
| 0.90 | 25/29 | 9.33333 | 82 | 8 | +0.04 s | 0.78 | 0.88 |
| 0.95 | 24.6667/29 | 7.33333 | 65 | 6.33333 | +0.06 s | 0.80 | 0.86 |
| 0.96 | 24.6667/29 | 6.66667 | 59 | 5.66667 | +0.07 s | 0.80 | 0.86 |
| 0.97 | 24/29 | 6.66667 | 59 | 5.66667 | +0.04 s | 0.80 | 0.85 |
| 0.98 | 24/29 | 6 | 53 | 5 | +0.07 s | 0.81 | 0.85 |
| 0.99 **(chosen)** | 24/29 | 5.66667 | 50 | 5 | +0.08 s | 0.82 | 0.84 |

## Fast-drop check (at the chosen threshold)

What requiring a fast body drop before an alert would change. The run's own setting is marked.

| Minimum drop (body lengths in 0.75 s) | Falls caught | False alerts |
|---:|---:|---:|
| off **(this run)** | 24/29 | 5.66667 |
| 0.05 | 24/29 | 5.66667 |
| 0.1 | 24/29 | 5.33333 |
| 0.15 | 24/29 | 5 |
| 0.2 | 24/29 | 4 |
| 0.25 | 23/29 | 2.33333 |
| 0.3 | 21/29 | 0.333333 |

## By batch (rig clips, at the chosen threshold)

`trial1` is exactly the clip set Trial 1 was scored on, so it compares directly with Trial 1's numbers. Later batches are the clips added since.

| Batch | Clips | Falls caught | False alerts | Non-fall minutes |
|---|---:|---:|---:|---:|
| trial1 | 31 | 19.6667/21 | 5.66667 | 5.8 |
| trial2 | 8 | 4.33333/8 | 0 | 1.0 |

## Folds

| Seed | Fold | Test clips | Train windows (falls) | Test windows (falls) | Best epoch | Avg precision | F1 @0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 11 | 14448 (1647) | 1947 (220) | 3 | 0.96 | 0.89 |
| 0 | 2 | 7 | 14856 (1727) | 1748 (113) | 15 | 0.83 | 0.75 |
| 0 | 3 | 7 | 14338 (1619) | 1948 (144) | 2 | 0.99 | 0.90 |
| 0 | 4 | 7 | 14349 (1634) | 1841 (128) | 7 | 0.74 | 0.70 |
| 0 | 5 | 7 | 14293 (1731) | 2289 (144) | 5 | 0.95 | 0.66 |
| 1 | 1 | 11 | 14517 (1628) | 1947 (220) | 13 | 0.87 | 0.87 |
| 1 | 2 | 7 | 14843 (1750) | 1748 (113) | 7 | 0.93 | 0.82 |
| 1 | 3 | 7 | 14661 (1731) | 1948 (144) | 10 | 0.92 | 0.84 |
| 1 | 4 | 7 | 14540 (1662) | 1841 (128) | 1 | 0.75 | 0.66 |
| 1 | 5 | 7 | 14507 (1727) | 2289 (144) | 40 | 0.97 | 0.85 |
| 2 | 1 | 11 | 14325 (1640) | 1947 (220) | 8 | 0.98 | 0.89 |
| 2 | 2 | 7 | 14371 (1649) | 1748 (113) | 5 | 0.93 | 0.87 |
| 2 | 3 | 7 | 14595 (1701) | 1948 (144) | 8 | 0.95 | 0.79 |
| 2 | 4 | 7 | 14433 (1712) | 1841 (128) | 3 | 0.97 | 0.89 |
| 2 | 5 | 7 | 14220 (1723) | 2289 (144) | 21 | 0.95 | 0.71 |

## Per clip (at the chosen threshold)

| Clip | Fold | Fall (s) | Fall windows | Caught | Alert time | False alerts | Peak p faller | Peak p others |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1xiFn4bLdT-6Alz7z0XDz_alligator | 5 | - | - | - | - | 4 | - | 1.00 |
| 29jB-BibMmaRAZBqZtNnU_alligator | 3 | - | - | - | - | 4 | - | 1.00 |
| 4CJH5B2NtzP8VwFzxMwO5_alligator | 1 | - | - | - | - | 0 | - | 0.32 |
| EIlywKhfkRHTAlBqul_mf_alligator | 2 | - | - | - | - | 1 | - | 0.78 |
| ILLOvByCSJmu2U4QL4R7q_output | 1 | 3.00-3.50 | 22 | 3/3 | +0.14 | 0 | 1.00 | - |
| LXCUWr1Hjb3MEZVgdYPIh_alligator | 2 | - | - | - | - | 0 | - | 0.74 |
| MIPR_SGTQICev4mFBDU2c_alligator | 3 | - | - | - | - | 0 | - | 0.56 |
| UyNpEpYTGgJkfrlSRc7Kd_alligator | 5 | - | - | - | - | 2 | - | 1.00 |
| YdQeOIiCjBFNpPjdTE641_output | 1 | 6.75-7.33 | 29 | 3/3 | +0.36 | 0 | 1.00 | - |
| _4RXOScjeGcGW1uLah2Cb_output | 1 | 8.83-9.50 | 29 | 3/3 | +0.25 | 0 | 1.00 | - |
| backward_fall_near_equipment | 4 | 9.75-10.75 | 30 | 3/3 | -0.36 | 0 | 1.00 | 0.12 |
| backward_slip_while_stepping_back | 5 | 10.00-10.67 | 26 | 3/3 | +0.36 | 0 | 1.00 | 0.02 |
| f40KfQHxg3C36Lj1rYwqk_alligator | 4 | - | - | - | - | 0 | - | 0.84 |
| fall_from_ladder | 1 | 7.50-8.08 | 25 | 3/3 | +0.08 | 0 | 1.00 | 0.08 |
| fall_while_working_on_pipe | 2 | 7.75-8.67 | 29 | 3/3 | -0.08 | 0 | 1.00 | 0.34 |
| forward_fall_while_carrying_a_tool | 5 | 8.58-9.25 | 30 | 3/3 | -0.06 | 0 | 1.00 | 0.62 |
| forward_trip_while_walking | 3 | 5.25-5.92 | 30 | 3/3 | -0.03 | 0 | 1.00 | 0.96 |
| jrigd_ZgdOhh4j67tzGVU_alligator | 1 | - | - | - | - | 4 | - | 1.00 |
| miller_group_fall_01_slip_wet_grating | 4 | 6.08-6.42 | 30 | 0/3 | - | 0 | 0.43 | 0.00 |
| miller_group_fall_02_trip_over_hose | 3 | 4.17-4.75 | 29 | 3/3 | +0.11 | 0 | 1.00 | 0.00 |
| miller_group_fall_03_slip_oily_patch_drizzle | 4 | 3.42-4.25 | 30 | 3/3 | +0.03 | 0 | 1.00 | 0.02 |
| miller_group_fall_04_ladder_fall_pipe_rack | 3 | 7.00-7.42 | 29 | 3/3 | +0.19 | 0 | 1.00 | 0.03 |
| miller_group_fall_05_stair_misstep_night | 4 | 5.75-6.50 | 29 | 2/3 | +0.12 | 0 | 0.97 | 0.00 |
| miller_group_fall_08_fall_from_low_platform | 1 | 4.67-5.25 | 29 | 3/3 | -0.08 | 0 | 1.00 | 0.00 |
| miller_group_fall_10_slip_carrying_toolbox_night_rain | 3 | 4.17-5.08 | 29 | 3/3 | -0.28 | 0 | 1.00 | 0.00 |
| miller_night_fall_01_nv_slip_wet_grating | 2 | 5.42-6.08 | 2 | 0/3 | - | 0 | 0.30 | 0.00 |
| miller_night_fall_02_nv_trip_over_hose | 1 | 4.25-4.83 | 0 | 0/3 | - | 0 | 0.08 | 0.41 |
| miller_night_fall_05_nv_stumble_carrying_pipe | 4 | 4.92-5.42 | 9 | 0/3 | - | 0 | 1.00 | 0.01 |
| miller_night_fall_06_oh_slip_oily_patch | 1 | 4.00-4.50 | 30 | 3/3 | -0.19 | 0 | 1.00 | 0.00 |
| miller_night_fall_07_oh_trip_over_flange | 2 | 4.08-4.50 | 28 | 1/3 | +0.42 | 0 | 1.00 | 0.17 |
| miller_night_fall_08_oh_fall_from_low_platform | 1 | 4.92-5.50 | 28 | 3/3 | +0.42 | 0 | 1.00 | 0.01 |
| miller_night_fall_09_oh_slip_stepping_off_ladder | 5 | 5.08-5.58 | 30 | 3/3 | -0.06 | 0 | 1.00 | 0.33 |
| miller_night_fall_10_oh_slip_carrying_toolbox_alone | 2 | 4.42-4.83 | 24 | 3/3 | +0.17 | 0 | 1.00 | 0.00 |
| oO8L7gEH7O6hbAxYdXVHD_output | 1 | 2.08-2.50 | 28 | 3/3 | +0.25 | 0 | 1.00 | 0.00 |
| slip_behind_person | 3 | 7.25-7.75 | 27 | 3/3 | +0.33 | 1 | 1.00 | 0.37 |
| slip_between_coworkers | 5 | 6.83-7.67 | 29 | 3/3 | -0.39 | 0 | 1.00 | 0.33 |
| slip_in_the_open | 2 | 6.33-6.67 | 30 | 3/3 | +0.47 | 1 | 1.00 | 0.69 |
| sudden_collapse | 5 | 6.25-6.67 | 29 | 3/3 | -0.11 | 0 | 1.00 | 0.08 |
| yQ3kWNc-Qe3EgqRaNso6l_alligator | 4 | - | - | - | - | 0 | - | 0.08 |

## Notes

- 1 fall clip(s) give no fall windows because the pose detector loses the faller mid-fall: miller_night_fall_02_nv_trip_over_hose. They can only be caught if the model fires on the onset.

## Plots

![curves](plots/curves.png)

![precision_recall](plots/precision_recall.png)

![thresholds](plots/thresholds.png)

