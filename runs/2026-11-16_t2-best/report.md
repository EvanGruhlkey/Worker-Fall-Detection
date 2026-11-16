# Temporal model run: t2-best

2026-11-16T08:11:39 · commit `3fa0f41+changes` · manifest.json: 161 clips, 101 falls

## Headline

At the chosen gate threshold **0.99** (mean of 3 seeds), held out by 5-fold grouped cross-validation (every clip tested once, by a model that never saw its scene):

- **Falls caught: 25 of 29** (86%), alert within 1.5 s of impact
- **False alerts: 4.33333** in 6.8 min of non-fall footage (4.33333 on the normal-work clips)
- Median alert time: +0.17 s relative to impact
- Late alerts (faller, after the catch window): 0.666667
- Window-level: average precision 0.88, precision 0.85 / recall 0.83 at this threshold
- **Honest estimate (threshold chosen without the tested fold): 25 of 29 caught, 4.66667 false alerts.** Each fold's threshold (0.90, 0.97, 0.98, 0.99) was picked from the other folds only, so these are the numbers to compare between runs.
- Seed to seed: falls caught 25 to 25, false alerts 3 to 6 (the run-to-run noise; differences between runs smaller than this are not real)

> One false alert in 6.8 min is 9/hour, so this data cannot show the 1/hour target directly; the threshold is chosen for the fewest false alerts seen, then the most falls caught.

## Data and settings

- 39 clips (29 with a fall), 36 scene groups, 11.9998 fps, windows of 30 frames (2.5 s), stride 1
- Model: GRU, hidden 64, dropout 0.1, up to 40 epochs (best by validation), learning rate 0.002
- Ambiguous windows (0-50% of a fall) left out: yes
- Augmentation: one mirrored copy per training clip; synthetic clips added: 0
- Label smoothing: 0; low-posture normal windows weighted x1
- Tracking: a lost track continues for up to 5 frames (0.42 s); duplicate boxes on one body removed
- Gate: debounce 5 frames (0.42 s), cooldown 30 s
- Fast-drop check: alerts need a body drop of at least 0.15 body lengths within 0.75 s
- Seeds: 0, 1, 2
- Tested clips by source: rig 39 (29 falls)
- External clips added to every training set (never tested): caucafall 88 (38 falls), gym 22 (22 falls)

## Threshold trade-off

| Threshold | Falls caught | False alerts | Per hour | Normal-clip false | Median alert time | Window precision | Window recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 25/29 | 12 | 106 | 8.66667 | -0.03 s | 0.66 | 0.93 |
| 0.10 | 25/29 | 11.6667 | 103 | 8.66667 | -0.03 s | 0.68 | 0.93 |
| 0.15 | 25/29 | 11.3333 | 100 | 8.33333 | +0.00 s | 0.69 | 0.92 |
| 0.20 | 25.3333/29 | 10.3333 | 91 | 8.33333 | +0.00 s | 0.70 | 0.92 |
| 0.25 | 25.3333/29 | 10 | 88 | 8.33333 | +0.00 s | 0.71 | 0.92 |
| 0.30 | 25/29 | 9.33333 | 82 | 7.66667 | +0.00 s | 0.71 | 0.91 |
| 0.35 | 25/29 | 9.33333 | 82 | 7.66667 | +0.00 s | 0.72 | 0.91 |
| 0.40 | 25/29 | 9.33333 | 82 | 7.66667 | +0.00 s | 0.72 | 0.91 |
| 0.45 | 25/29 | 9 | 79 | 7.33333 | +0.00 s | 0.73 | 0.91 |
| 0.50 | 25.3333/29 | 8.66667 | 76 | 7.33333 | +0.00 s | 0.74 | 0.91 |
| 0.55 | 25.3333/29 | 8.33333 | 74 | 7 | +0.00 s | 0.74 | 0.91 |
| 0.60 | 25.3333/29 | 8.33333 | 74 | 7 | +0.00 s | 0.75 | 0.91 |
| 0.65 | 25.3333/29 | 7.66667 | 68 | 6.66667 | +0.00 s | 0.75 | 0.90 |
| 0.70 | 25.3333/29 | 7.66667 | 68 | 6.66667 | +0.03 s | 0.76 | 0.90 |
| 0.75 | 25.3333/29 | 7.33333 | 65 | 6.33333 | +0.04 s | 0.76 | 0.90 |
| 0.80 | 25.3333/29 | 7.33333 | 65 | 6.33333 | +0.04 s | 0.77 | 0.89 |
| 0.85 | 25.3333/29 | 7 | 62 | 6.33333 | +0.06 s | 0.78 | 0.89 |
| 0.90 | 25.3333/29 | 6.33333 | 56 | 6 | +0.08 s | 0.79 | 0.88 |
| 0.95 | 25.3333/29 | 6 | 53 | 5.66667 | +0.08 s | 0.81 | 0.87 |
| 0.96 | 25.3333/29 | 6 | 53 | 5.66667 | +0.08 s | 0.82 | 0.86 |
| 0.97 | 25.3333/29 | 5.66667 | 50 | 5.33333 | +0.12 s | 0.82 | 0.86 |
| 0.98 | 25.3333/29 | 5.66667 | 50 | 5.33333 | +0.17 s | 0.83 | 0.85 |
| 0.99 **(chosen)** | 25/29 | 4.33333 | 38 | 4.33333 | +0.17 s | 0.85 | 0.83 |

## Fast-drop check (at the chosen threshold)

What requiring a fast body drop before an alert would change. The run's own setting is marked.

| Minimum drop (body lengths in 0.75 s) | Falls caught | False alerts |
|---:|---:|---:|
| off | 25/29 | 5 |
| 0.05 | 25/29 | 4.66667 |
| 0.1 | 25/29 | 4.66667 |
| 0.15 **(this run)** | 25/29 | 4.33333 |
| 0.2 | 24/29 | 4 |
| 0.25 | 23/29 | 2.66667 |
| 0.3 | 20.6667/29 | 0.333333 |

## By batch (rig clips, at the chosen threshold)

`trial1` is exactly the clip set Trial 1 was scored on, so it compares directly with Trial 1's numbers. Later batches are the clips added since.

| Batch | Clips | Falls caught | False alerts | Non-fall minutes |
|---|---:|---:|---:|---:|
| trial1 | 31 | 20.3333/21 | 4.33333 | 5.8 |
| trial2 | 8 | 4.66667/8 | 0 | 1.0 |

## Folds

| Seed | Fold | Test clips | Train windows (falls) | Test windows (falls) | Best epoch | Avg precision | F1 @0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 11 | 28896 (3294) | 1947 (220) | 1 | 0.92 | 0.78 |
| 0 | 2 | 7 | 29712 (3454) | 1748 (113) | 5 | 0.63 | 0.55 |
| 0 | 3 | 7 | 28676 (3238) | 1948 (144) | 7 | 0.87 | 0.88 |
| 0 | 4 | 7 | 28698 (3268) | 1841 (128) | 16 | 0.97 | 0.91 |
| 0 | 5 | 7 | 28586 (3462) | 2289 (144) | 2 | 0.98 | 0.74 |
| 1 | 1 | 11 | 29034 (3256) | 1947 (220) | 7 | 0.87 | 0.86 |
| 1 | 2 | 7 | 29686 (3500) | 1748 (113) | 7 | 0.89 | 0.88 |
| 1 | 3 | 7 | 29322 (3462) | 1948 (144) | 4 | 0.98 | 0.93 |
| 1 | 4 | 7 | 29080 (3324) | 1841 (128) | 21 | 0.98 | 0.91 |
| 1 | 5 | 7 | 29014 (3454) | 2289 (144) | 15 | 0.87 | 0.73 |
| 2 | 1 | 11 | 28650 (3280) | 1947 (220) | 15 | 0.92 | 0.85 |
| 2 | 2 | 7 | 28742 (3298) | 1748 (113) | 9 | 0.77 | 0.72 |
| 2 | 3 | 7 | 29190 (3402) | 1948 (144) | 9 | 0.88 | 0.72 |
| 2 | 4 | 7 | 28866 (3424) | 1841 (128) | 6 | 1.00 | 0.99 |
| 2 | 5 | 7 | 28440 (3446) | 2289 (144) | 5 | 0.88 | 0.77 |

## Per clip (at the chosen threshold)

| Clip | Fold | Fall (s) | Fall windows | Caught | Alert time | False alerts | Peak p faller | Peak p others |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1xiFn4bLdT-6Alz7z0XDz_alligator | 5 | - | - | - | - | 4 | - | 1.00 |
| 29jB-BibMmaRAZBqZtNnU_alligator | 3 | - | - | - | - | 3 | - | 1.00 |
| 4CJH5B2NtzP8VwFzxMwO5_alligator | 1 | - | - | - | - | 2 | - | 0.79 |
| EIlywKhfkRHTAlBqul_mf_alligator | 2 | - | - | - | - | 0 | - | 0.70 |
| ILLOvByCSJmu2U4QL4R7q_output | 1 | 3.00-3.50 | 22 | 3/3 | +0.31 | 0 | 1.00 | - |
| LXCUWr1Hjb3MEZVgdYPIh_alligator | 2 | - | - | - | - | 0 | - | 0.30 |
| MIPR_SGTQICev4mFBDU2c_alligator | 3 | - | - | - | - | 0 | - | 0.24 |
| UyNpEpYTGgJkfrlSRc7Kd_alligator | 5 | - | - | - | - | 1 | - | 0.95 |
| YdQeOIiCjBFNpPjdTE641_output | 1 | 6.75-7.33 | 29 | 3/3 | +0.36 | 0 | 1.00 | - |
| _4RXOScjeGcGW1uLah2Cb_output | 1 | 8.83-9.50 | 29 | 3/3 | +0.25 | 0 | 1.00 | - |
| backward_fall_near_equipment | 4 | 9.75-10.75 | 30 | 3/3 | -0.33 | 0 | 1.00 | 0.01 |
| backward_slip_while_stepping_back | 5 | 10.00-10.67 | 26 | 3/3 | +0.39 | 0 | 1.00 | 0.05 |
| f40KfQHxg3C36Lj1rYwqk_alligator | 4 | - | - | - | - | 0 | - | 0.11 |
| fall_from_ladder | 1 | 7.50-8.08 | 25 | 3/3 | +0.03 | 0 | 1.00 | 0.84 |
| fall_while_working_on_pipe | 2 | 7.75-8.67 | 29 | 3/3 | -0.22 | 0 | 1.00 | 0.05 |
| forward_fall_while_carrying_a_tool | 5 | 8.58-9.25 | 30 | 3/3 | +0.06 | 0 | 1.00 | 0.34 |
| forward_trip_while_walking | 3 | 5.25-5.92 | 30 | 3/3 | +0.00 | 0 | 1.00 | 0.51 |
| jrigd_ZgdOhh4j67tzGVU_alligator | 1 | - | - | - | - | 3 | - | 1.00 |
| miller_group_fall_01_slip_wet_grating | 4 | 6.08-6.42 | 30 | 3/3 | +0.47 | 0 | 1.00 | 0.00 |
| miller_group_fall_02_trip_over_hose | 3 | 4.17-4.75 | 29 | 3/3 | +0.17 | 0 | 1.00 | 0.00 |
| miller_group_fall_03_slip_oily_patch_drizzle | 4 | 3.42-4.25 | 30 | 3/3 | -0.14 | 0 | 1.00 | 0.00 |
| miller_group_fall_04_ladder_fall_pipe_rack | 3 | 7.00-7.42 | 29 | 3/3 | +0.11 | 0 | 1.00 | 0.00 |
| miller_group_fall_05_stair_misstep_night | 4 | 5.75-6.50 | 29 | 2/3 | +0.12 | 0 | 1.00 | 0.00 |
| miller_group_fall_08_fall_from_low_platform | 1 | 4.67-5.25 | 29 | 3/3 | -0.08 | 0 | 1.00 | 0.00 |
| miller_group_fall_10_slip_carrying_toolbox_night_rain | 3 | 4.17-5.08 | 29 | 3/3 | +0.03 | 0 | 1.00 | 0.00 |
| miller_night_fall_01_nv_slip_wet_grating | 2 | 5.42-6.08 | 2 | 0/3 | - | 0 | 1.00 | 0.00 |
| miller_night_fall_02_nv_trip_over_hose | 1 | 4.25-4.83 | 0 | 0/3 | - | 0 | 0.14 | 0.66 |
| miller_night_fall_05_nv_stumble_carrying_pipe | 4 | 4.92-5.42 | 9 | 1/3 | +0.25 | 0 | 1.00 | 0.33 |
| miller_night_fall_06_oh_slip_oily_patch | 1 | 4.00-4.50 | 30 | 3/3 | -0.11 | 0 | 1.00 | 0.01 |
| miller_night_fall_07_oh_trip_over_flange | 2 | 4.08-4.50 | 28 | 1/3 | +0.75 | 0 | 1.00 | 0.12 |
| miller_night_fall_08_oh_fall_from_low_platform | 1 | 4.92-5.50 | 28 | 3/3 | +0.33 | 0 | 1.00 | 0.26 |
| miller_night_fall_09_oh_slip_stepping_off_ladder | 5 | 5.08-5.58 | 30 | 3/3 | +0.19 | 0 | 1.00 | 0.42 |
| miller_night_fall_10_oh_slip_carrying_toolbox_alone | 2 | 4.42-4.83 | 24 | 3/3 | +0.36 | 0 | 1.00 | 0.00 |
| oO8L7gEH7O6hbAxYdXVHD_output | 1 | 2.08-2.50 | 28 | 3/3 | +0.25 | 0 | 1.00 | 0.01 |
| slip_behind_person | 3 | 7.25-7.75 | 27 | 2/3 | +0.67 | 0 | 0.97 | 0.66 |
| slip_between_coworkers | 5 | 6.83-7.67 | 29 | 3/3 | +0.50 | 0 | 1.00 | 0.19 |
| slip_in_the_open | 2 | 6.33-6.67 | 30 | 3/3 | +1.17 | 0 | 1.00 | 0.10 |
| sudden_collapse | 5 | 6.25-6.67 | 29 | 3/3 | +0.00 | 0 | 1.00 | 0.74 |
| yQ3kWNc-Qe3EgqRaNso6l_alligator | 4 | - | - | - | - | 0 | - | 0.00 |

## Notes

- 1 fall clip(s) give no fall windows because the pose detector loses the faller mid-fall: miller_night_fall_02_nv_trip_over_hose. They can only be caught if the model fires on the onset.

## Plots

![curves](plots/curves.png)

![precision_recall](plots/precision_recall.png)

![thresholds](plots/thresholds.png)

