# Temporal model run: plus-cauca

2026-11-05T20:22:08 · commit `897c565+changes` · manifest.json: 131 clips, 71 falls

## Headline

At the chosen gate threshold **0.99** (mean of 2 seeds), held out by 5-fold grouped cross-validation (every clip tested once, by a model that never saw its scene):

- **Falls caught: 13.5 of 21** (64%), alert within 1.5 s of impact
- **False alerts: 6** in 5.8 min of non-fall footage (4 on the normal-work clips)
- Median alert time: +0.17 s relative to impact
- Late alerts (faller, after the catch window): 0
- Window-level: average precision 0.73, precision 0.73 / recall 0.71 at this threshold

> One false alert in 5.8 min is 10/hour, so this data cannot show the 1/hour target directly; the threshold is chosen for the fewest false alerts seen, then the most falls caught.

## Data and settings

- 31 clips (21 with a fall), 28 scene groups, 12 fps, windows of 30 frames (2.5 s), stride 1
- Model: GRU, hidden 64, dropout 0.1, up to 40 epochs (best by validation), learning rate 0.002
- Ambiguous windows (0-50% of a fall) left out: yes
- Augmented copies per training clip: 0; synthetic clips added: 0
- Gate: debounce 5 frames (0.42 s), cooldown 30 s
- Seeds: 0, 1
- Tested clips by source: rig 31 (21 falls)
- External clips added to every training set (never tested): caucafall 100 (50 falls)

## Threshold trade-off

| Threshold | Falls caught | False alerts | Per hour | Normal-clip false | Median alert time | Window precision | Window recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 12.5/21 | 17.5 | 181 | 11.5 | -0.08 s | 0.43 | 0.83 |
| 0.10 | 12.5/21 | 17 | 176 | 11.5 | -0.04 s | 0.46 | 0.83 |
| 0.15 | 12.5/21 | 16.5 | 171 | 11.5 | -0.02 s | 0.48 | 0.82 |
| 0.20 | 12.5/21 | 16.5 | 171 | 11.5 | -0.02 s | 0.49 | 0.82 |
| 0.25 | 12.5/21 | 15.5 | 161 | 10.5 | -0.02 s | 0.50 | 0.81 |
| 0.30 | 12.5/21 | 15.5 | 161 | 10.5 | +0.06 s | 0.50 | 0.81 |
| 0.35 | 12.5/21 | 15.5 | 161 | 10.5 | +0.06 s | 0.50 | 0.80 |
| 0.40 | 12.5/21 | 14.5 | 150 | 9.5 | +0.06 s | 0.51 | 0.80 |
| 0.45 | 12.5/21 | 14 | 145 | 9 | +0.06 s | 0.52 | 0.80 |
| 0.50 | 13/21 | 13.5 | 140 | 9 | +0.04 s | 0.53 | 0.80 |
| 0.55 | 13/21 | 13 | 135 | 9 | +0.04 s | 0.54 | 0.79 |
| 0.60 | 13/21 | 13 | 135 | 9 | +0.04 s | 0.55 | 0.79 |
| 0.65 | 13/21 | 12.5 | 130 | 8.5 | +0.04 s | 0.55 | 0.79 |
| 0.70 | 13/21 | 12.5 | 130 | 8.5 | +0.04 s | 0.57 | 0.78 |
| 0.75 | 13/21 | 12.5 | 130 | 8.5 | +0.04 s | 0.58 | 0.77 |
| 0.80 | 13.5/21 | 11.5 | 119 | 8 | +0.06 s | 0.59 | 0.76 |
| 0.85 | 13.5/21 | 10.5 | 109 | 7 | +0.08 s | 0.61 | 0.76 |
| 0.90 | 13.5/21 | 9 | 93 | 5.5 | +0.08 s | 0.65 | 0.76 |
| 0.95 | 13/21 | 8 | 83 | 5 | +0.17 s | 0.69 | 0.74 |
| 0.96 | 13/21 | 7.5 | 78 | 4.5 | +0.17 s | 0.70 | 0.74 |
| 0.97 | 13.5/21 | 7 | 73 | 4.5 | +0.17 s | 0.71 | 0.73 |
| 0.98 | 13.5/21 | 7 | 73 | 4.5 | +0.17 s | 0.72 | 0.73 |
| 0.99 **(chosen)** | 13.5/21 | 6 | 62 | 4 | +0.17 s | 0.73 | 0.71 |

## Folds

| Seed | Fold | Test clips | Train windows (falls) | Test windows (falls) | Best epoch | Avg precision | F1 @0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 9 | 12268 (875) | 1552 (113) | 7 | 0.97 | 0.67 |
| 0 | 2 | 6 | 12430 (922) | 1397 (87) | 21 | 0.34 | 0.32 |
| 0 | 3 | 6 | 11692 (875) | 2091 (112) | 7 | 0.96 | 0.80 |
| 0 | 4 | 5 | 11685 (977) | 1512 (34) | 2 | 0.75 | 0.71 |
| 0 | 5 | 5 | 11985 (902) | 1797 (87) | 11 | 0.67 | 0.65 |
| 1 | 1 | 9 | 12439 (903) | 1552 (113) | 3 | 0.96 | 0.58 |
| 1 | 2 | 6 | 11871 (898) | 1397 (87) | 2 | 0.62 | 0.53 |
| 1 | 3 | 6 | 11198 (845) | 2091 (112) | 25 | 0.84 | 0.82 |
| 1 | 4 | 5 | 11649 (952) | 1512 (34) | 6 | 0.90 | 0.78 |
| 1 | 5 | 5 | 11977 (930) | 1797 (87) | 13 | 0.66 | 0.68 |

## Per clip (at the chosen threshold)

| Clip | Fold | Fall (s) | Fall windows | Caught | Alert time | False alerts | Peak p faller | Peak p others |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1xiFn4bLdT-6Alz7z0XDz_alligator | 2 | - | - | - | - | 2 | - | 0.98 |
| 29jB-BibMmaRAZBqZtNnU_alligator | 5 | - | - | - | - | 0 | - | 1.00 |
| 4CJH5B2NtzP8VwFzxMwO5_alligator | 4 | - | - | - | - | 0 | - | 0.97 |
| EIlywKhfkRHTAlBqul_mf_alligator | 3 | - | - | - | - | 1 | - | 0.99 |
| ILLOvByCSJmu2U4QL4R7q_output | 1 | 3.00-3.50 | 0 | 0/2 | - | 0 | 0.69 | - |
| LXCUWr1Hjb3MEZVgdYPIh_alligator | 5 | - | - | - | - | 0 | - | 0.00 |
| MIPR_SGTQICev4mFBDU2c_alligator | 4 | - | - | - | - | 0 | - | 0.56 |
| UyNpEpYTGgJkfrlSRc7Kd_alligator | 1 | - | - | - | - | 2 | - | 1.00 |
| YdQeOIiCjBFNpPjdTE641_output | 1 | 6.75-7.33 | 28 | 2/2 | +0.54 | 0 | 1.00 | 0.00 |
| _4RXOScjeGcGW1uLah2Cb_output | 1 | 8.83-9.50 | 8 | 2/2 | +0.25 | 0 | 1.00 | 0.45 |
| backward_fall_near_equipment | 5 | 9.75-10.75 | 30 | 2/2 | -0.38 | 0 | 1.00 | 0.99 |
| backward_slip_while_stepping_back | 2 | 10.00-10.67 | 3 | 0/2 | - | 0 | 0.91 | 0.12 |
| f40KfQHxg3C36Lj1rYwqk_alligator | 2 | - | - | - | - | 0 | - | 0.99 |
| fall_from_ladder | 3 | 7.50-8.08 | 27 | 2/2 | +0.25 | 0 | 1.00 | 0.27 |
| fall_while_working_on_pipe | 1 | 7.75-8.67 | 27 | 2/2 | +0.04 | 1 | 1.00 | 0.61 |
| forward_fall_while_carrying_a_tool | 3 | 8.58-9.25 | 30 | 2/2 | +0.08 | 0 | 1.00 | 0.02 |
| forward_trip_while_walking | 3 | 5.25-5.92 | 28 | 2/2 | +0.00 | 0 | 1.00 | 0.01 |
| jrigd_ZgdOhh4j67tzGVU_alligator | 3 | - | - | - | - | 2 | - | 1.00 |
| miller_group_fall_01_slip_wet_grating | 2 | 6.08-6.42 | 30 | 0/2 | - | 0 | 0.97 | 0.00 |
| miller_group_fall_02_trip_over_hose | 1 | 4.17-4.75 | 26 | 2/2 | +0.42 | 0 | 1.00 | 0.01 |
| miller_group_fall_03_slip_oily_patch_drizzle | 4 | 3.42-4.25 | 8 | 0/2 | - | 0 | 0.65 | 0.05 |
| miller_group_fall_04_ladder_fall_pipe_rack | 2 | 7.00-7.42 | 26 | 2/2 | +0.08 | 1 | 1.00 | 1.00 |
| miller_group_fall_05_stair_misstep_night | 2 | 5.75-6.50 | 28 | 1/2 | +0.08 | 0 | 1.00 | 0.00 |
| miller_group_fall_08_fall_from_low_platform | 4 | 4.67-5.25 | 0 | 0/2 | - | 0 | 0.40 | 0.00 |
| miller_group_fall_10_slip_carrying_toolbox_night_rain | 5 | 4.17-5.08 | 29 | 0/2 | - | 0 | 0.50 | - |
| oO8L7gEH7O6hbAxYdXVHD_output | 1 | 2.08-2.50 | 24 | 2/2 | +0.58 | 0 | 1.00 | 0.00 |
| slip_behind_person | 1 | 7.25-7.75 | 0 | 0/2 | - | 2 | 0.01 | 1.00 |
| slip_between_coworkers | 3 | 6.83-7.67 | 27 | 2/2 | -0.42 | 0 | 1.00 | 0.98 |
| slip_in_the_open | 4 | 6.33-6.67 | 26 | 2/2 | +0.62 | 0 | 1.00 | 0.52 |
| sudden_collapse | 5 | 6.25-6.67 | 28 | 2/2 | -0.00 | 0 | 1.00 | 0.07 |
| yQ3kWNc-Qe3EgqRaNso6l_alligator | 1 | - | - | - | - | 1 | - | 1.00 |

## Notes

- 3 fall clip(s) give no fall windows because the pose detector loses the faller mid-fall: ILLOvByCSJmu2U4QL4R7q_output, miller_group_fall_08_fall_from_low_platform, slip_behind_person. They can only be caught if the model fires on the onset.

## Plots

![curves](plots/curves.png)

![precision_recall](plots/precision_recall.png)

![thresholds](plots/thresholds.png)

![timelines_falls](plots/timelines_falls.png)

![timelines_normal](plots/timelines_normal.png)

