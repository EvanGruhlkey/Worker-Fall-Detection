# Temporal model run: t3-base

2026-11-19T19:51:39 · commit `2f4e79d+changes` · manifest.json: 237 clips, 114 falls

## Headline

At the chosen gate threshold **0.99** (mean of 3 seeds), held out by 5-fold grouped cross-validation (every clip tested once, by a model that never saw its scene):

- **Falls caught: 34.3333 of 42** (82%), alert within 1.5 s of impact
- **False alerts: 9** in 20.7 min of non-fall footage (8.66667 on the normal-work clips)
- Median alert time: +0.22 s relative to impact
- Late alerts (faller, after the catch window): 2
- Window-level: average precision 0.80, precision 0.81 / recall 0.81 at this threshold
- **Honest estimate (threshold chosen without the tested fold): 34.3333 of 42 caught, 10.3333 false alerts.** Each fold's threshold (0.30, 0.97, 0.99) was picked from the other folds only, so these are the numbers to compare between runs.
- Seed to seed: falls caught 33 to 35, false alerts 7 to 12 (the run-to-run noise; differences between runs smaller than this are not real)

> One false alert in 20.7 min is 3/hour, so this data cannot show the 1/hour target directly; the threshold is chosen for the fewest false alerts seen, then the most falls caught.

## Data and settings

- 115 clips (42 with a fall), 56 scene groups, 11.9998 fps, windows of 30 frames (2.5 s), stride 1
- Model: GRU, hidden 64, dropout 0.1, up to 40 epochs (best by validation), learning rate 0.002
- Ambiguous windows (0-50% of a fall) left out: yes
- Augmentation: one mirrored copy per training clip; synthetic clips added: 0
- Label smoothing: 0; low-posture normal windows weighted x1
- Tracking: a lost track continues for up to 5 frames (0.42 s); duplicate boxes on one body removed
- Gate: debounce 5 frames (0.42 s), cooldown 30 s
- Fast-drop check: alerts need a body drop of at least 0.15 body lengths within 0.75 s
- Seeds: 0, 1, 2
- Tested clips by source: rig 115 (42 falls)
- External clips added to every training set (never tested): caucafall 88 (38 falls), gym 22 (22 falls)

## Threshold trade-off

| Threshold | Falls caught | False alerts | Per hour | Normal-clip false | Median alert time | Window precision | Window recall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 37.3333/42 | 17 | 49 | 16 | +0.08 s | 0.71 | 0.89 |
| 0.10 | 37.3333/42 | 15.6667 | 45 | 15 | +0.08 s | 0.72 | 0.89 |
| 0.15 | 36.6667/42 | 15.3333 | 44 | 14.6667 | +0.08 s | 0.73 | 0.88 |
| 0.20 | 36.3333/42 | 15 | 44 | 14.3333 | +0.10 s | 0.74 | 0.88 |
| 0.25 | 36/42 | 15 | 44 | 14.3333 | +0.11 s | 0.74 | 0.88 |
| 0.30 | 36/42 | 14.6667 | 43 | 14 | +0.11 s | 0.75 | 0.88 |
| 0.35 | 35.6667/42 | 14 | 41 | 13.3333 | +0.11 s | 0.75 | 0.88 |
| 0.40 | 35.3333/42 | 14 | 41 | 13.3333 | +0.11 s | 0.75 | 0.87 |
| 0.45 | 35.3333/42 | 13 | 38 | 12.3333 | +0.11 s | 0.76 | 0.87 |
| 0.50 | 35.3333/42 | 12 | 35 | 11.3333 | +0.11 s | 0.76 | 0.87 |
| 0.55 | 35.3333/42 | 12 | 35 | 11.3333 | +0.11 s | 0.76 | 0.87 |
| 0.60 | 35.3333/42 | 11.3333 | 33 | 11 | +0.11 s | 0.77 | 0.87 |
| 0.65 | 35.3333/42 | 11.3333 | 33 | 11 | +0.11 s | 0.77 | 0.86 |
| 0.70 | 35.3333/42 | 11.3333 | 33 | 11 | +0.11 s | 0.77 | 0.86 |
| 0.75 | 35.3333/42 | 11.3333 | 33 | 11 | +0.11 s | 0.78 | 0.86 |
| 0.80 | 35.3333/42 | 11 | 32 | 10.6667 | +0.14 s | 0.78 | 0.86 |
| 0.85 | 35.3333/42 | 11 | 32 | 10.6667 | +0.14 s | 0.78 | 0.86 |
| 0.90 | 35.3333/42 | 10 | 29 | 9.66667 | +0.14 s | 0.79 | 0.85 |
| 0.95 | 35.3333/42 | 9.66667 | 28 | 9.33333 | +0.18 s | 0.80 | 0.84 |
| 0.96 | 35.3333/42 | 9.66667 | 28 | 9.33333 | +0.19 s | 0.80 | 0.84 |
| 0.97 | 35.3333/42 | 9.66667 | 28 | 9.33333 | +0.19 s | 0.80 | 0.83 |
| 0.98 | 34.3333/42 | 9.33333 | 27 | 9 | +0.19 s | 0.80 | 0.83 |
| 0.99 **(chosen)** | 34.3333/42 | 9 | 26 | 8.66667 | +0.22 s | 0.81 | 0.81 |

## Fast-drop check (at the chosen threshold)

What requiring a fast body drop before an alert would change. The run's own setting is marked.

| Minimum drop (body lengths in 0.75 s) | Falls caught | False alerts |
|---:|---:|---:|
| off | 34.6667/42 | 11.3333 |
| 0.05 | 34.6667/42 | 10.3333 |
| 0.1 | 34.6667/42 | 9.66667 |
| 0.15 **(this run)** | 34.3333/42 | 9 |
| 0.2 | 33.3333/42 | 9 |
| 0.25 | 28.6667/42 | 5 |
| 0.3 | 26/42 | 2.66667 |

## By batch (rig clips, at the chosen threshold)

`trial1` is exactly the clip set Trial 1 was scored on, so it compares directly with Trial 1's numbers. Later batches are the clips added since.

| Batch | Clips | Falls caught | False alerts | Non-fall minutes |
|---|---:|---:|---:|---:|
| trial1 | 31 | 20/21 | 2 | 5.8 |
| trial2 | 8 | 4.33333/8 | 0 | 1.0 |
| trial3 | 76 | 10/13 | 7 | 13.9 |

## False alerts by action (rig clips, at the chosen threshold)

Normal-work clips grouped by what they show (the action column of clips.csv). Clips without an action label are the Trial 1-2 normal-work clips; alerts on fall clips are those outside the fall (on coworkers, or before or after it). Per hour = false alerts / non-fall time.

| Action | Clips | False alerts | Non-fall minutes | Per hour |
|---|---:|---:|---:|---:|
| kneel_then_lie | 20 | 2.66667 | 5.0 | 32 |
| normal work, Trials 1-2 | 10 | 1.66667 | 2.3 | 43 |
| fast_getdown | 10 | 1.33333 | 1.7 | 48 |
| lying | 7 | 1.33333 | 1.2 | 68 |
| crouching | 5 | 1 | 0.8 | 71 |
| fall clips (outside the fall) | 42 | 0.333333 | 6.1 | 3 |
| ordinary | 2 | 0.333333 | 0.3 | 60 |
| sitting | 2 | 0.333333 | 0.3 | 60 |
| bending | 2 | 0 | 0.3 | 0 |
| climbing | 3 | 0 | 0.5 | 0 |
| crowd | 1 | 0 | 0.2 | 0 |
| frame_edge | 3 | 0 | 0.5 | 0 |
| kneeling | 4 | 0 | 0.7 | 0 |
| near_miss | 4 | 0 | 0.7 | 0 |

## Folds

| Seed | Fold | Test clips | Train windows (falls) | Test windows (falls) | Best epoch | Avg precision | F1 @0.5 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 17 | 51494 (3966) | 4649 (172) | 19 | 0.74 | 0.80 |
| 0 | 2 | 22 | 50096 (3904) | 5711 (232) | 40 | 0.80 | 0.80 |
| 0 | 3 | 33 | 49558 (3590) | 6129 (285) | 4 | 0.89 | 0.83 |
| 0 | 4 | 22 | 49862 (3926) | 4809 (226) | 30 | 0.67 | 0.81 |
| 0 | 5 | 21 | 50150 (3900) | 4657 (179) | 22 | 0.92 | 0.81 |
| 1 | 1 | 17 | 51072 (3966) | 4649 (172) | 23 | 0.73 | 0.83 |
| 1 | 2 | 22 | 46452 (3800) | 5711 (232) | 7 | 0.76 | 0.75 |
| 1 | 3 | 33 | 51150 (3840) | 6129 (285) | 21 | 0.88 | 0.82 |
| 1 | 4 | 22 | 52054 (3868) | 4809 (226) | 6 | 0.84 | 0.88 |
| 1 | 5 | 21 | 53314 (4020) | 4657 (179) | 8 | 0.92 | 0.76 |
| 2 | 1 | 17 | 54704 (3974) | 4649 (172) | 4 | 0.74 | 0.79 |
| 2 | 2 | 22 | 47470 (3898) | 5711 (232) | 3 | 0.91 | 0.89 |
| 2 | 3 | 33 | 50110 (3744) | 6129 (285) | 33 | 0.82 | 0.80 |
| 2 | 4 | 22 | 50960 (3858) | 4809 (226) | 26 | 0.90 | 0.86 |
| 2 | 5 | 21 | 47412 (3958) | 4657 (179) | 6 | 0.85 | 0.72 |

## Per clip (at the chosen threshold)

| Clip | Fold | Fall (s) | Fall windows | Caught | Alert time | False alerts | Peak p faller | Peak p others |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 1xiFn4bLdT-6Alz7z0XDz_alligator | 2 | - | - | - | - | 0 | - | 0.81 |
| 29jB-BibMmaRAZBqZtNnU_alligator | 1 | - | - | - | - | 1 | - | 0.83 |
| 4CJH5B2NtzP8VwFzxMwO5_alligator | 5 | - | - | - | - | 0 | - | 0.84 |
| EIlywKhfkRHTAlBqul_mf_alligator | 1 | - | - | - | - | 0 | - | 0.65 |
| ILLOvByCSJmu2U4QL4R7q_output | 4 | 3.00-3.50 | 22 | 3/3 | +0.25 | 0 | 1.00 | - |
| LXCUWr1Hjb3MEZVgdYPIh_alligator | 4 | - | - | - | - | 0 | - | 0.08 |
| MIPR_SGTQICev4mFBDU2c_alligator | 1 | - | - | - | - | 0 | - | 0.05 |
| UyNpEpYTGgJkfrlSRc7Kd_alligator | 3 | - | - | - | - | 0 | - | 0.40 |
| YdQeOIiCjBFNpPjdTE641_output | 4 | 6.75-7.33 | 29 | 3/3 | +0.39 | 0 | 1.00 | - |
| _4RXOScjeGcGW1uLah2Cb_output | 4 | 8.83-9.50 | 29 | 3/3 | +0.17 | 0 | 1.00 | - |
| backward_fall_near_equipment | 1 | 9.75-10.75 | 30 | 3/3 | -0.39 | 0 | 1.00 | 0.03 |
| backward_slip_while_stepping_back | 5 | 10.00-10.67 | 26 | 3/3 | +0.39 | 0 | 1.00 | 0.00 |
| f40KfQHxg3C36Lj1rYwqk_alligator | 2 | - | - | - | - | 0 | - | 0.34 |
| fall_from_ladder | 5 | 7.50-8.08 | 25 | 3/3 | -0.03 | 0 | 1.00 | 0.70 |
| fall_t2_01_slip_grating_backward_wet | 3 | 4.50-5.00 | 30 | 0/3 | - | 0 | 0.67 | 0.00 |
| fall_t2_02_slip_oil_sideways | 3 | 4.58-5.17 | 29 | 3/3 | +0.42 | 0 | 1.00 | 0.00 |
| fall_t2_03_slip_hose_forward_night | 5 | 3.33-3.92 | 29 | 3/3 | +0.58 | 0 | 1.00 | 0.00 |
| fall_t2_05_trip_cable_toward_camera | 3 | 4.83-5.33 | 30 | 3/3 | +0.31 | 0 | 1.00 | 0.00 |
| fall_t2_06_trip_grating_edge_dawn | 2 | 5.92-6.42 | 29 | 3/3 | +0.14 | 0 | 1.00 | 0.00 |
| fall_t2_07_trip_pipe_sling_across | 2 | 3.33-4.00 | 30 | 3/3 | +0.08 | 0 | 1.00 | 0.00 |
| fall_t2_08_backward_valve_wheel_gives | 2 | 4.83-5.33 | 30 | 3/3 | +0.97 | 0 | 1.00 | 0.00 |
| fall_t2_09_backward_pulling_hose_drizzle | 3 | 6.42-6.75 | 29 | 3/3 | +0.22 | 0 | 1.00 | 0.00 |
| fall_t2_10_backward_trip_toolbox_night | 1 | 4.75-5.17 | 0 | 0/3 | - | 0 | 0.06 | 0.01 |
| fall_t2_11_sudden_collapse_at_panel | 3 | 4.17-4.75 | 29 | 3/3 | +0.31 | 0 | 1.00 | 0.00 |
| fall_t2_12_knocked_by_swinging_pipe | 3 | 5.42-6.00 | 29 | 3/3 | +0.19 | 0 | 1.00 | 0.00 |
| fall_t2_13_fall_behind_pump_partial | 3 | 6.50-7.00 | 22 | 0/3 | - | 0 | 0.98 | 0.00 |
| fall_t2_15_fall_from_platform_night_rain | 2 | 5.17-5.83 | 29 | 3/3 | -0.11 | 0 | 1.00 | 0.00 |
| fall_while_working_on_pipe | 1 | 7.75-8.67 | 29 | 3/3 | -0.33 | 0 | 1.00 | 0.00 |
| forward_fall_while_carrying_a_tool | 1 | 8.58-9.25 | 30 | 3/3 | +0.36 | 0 | 1.00 | 0.00 |
| forward_trip_while_walking | 1 | 5.25-5.92 | 30 | 3/3 | -0.08 | 0 | 1.00 | 0.64 |
| jrigd_ZgdOhh4j67tzGVU_alligator | 5 | - | - | - | - | 4 | - | 1.00 |
| miller_group_fall_01_slip_wet_grating | 5 | 6.08-6.42 | 30 | 3/3 | +0.64 | 0 | 1.00 | 0.00 |
| miller_group_fall_02_trip_over_hose | 4 | 4.17-4.75 | 29 | 3/3 | +0.25 | 0 | 1.00 | 0.00 |
| miller_group_fall_03_slip_oily_patch_drizzle | 5 | 3.42-4.25 | 30 | 3/3 | -0.22 | 0 | 1.00 | 0.00 |
| miller_group_fall_04_ladder_fall_pipe_rack | 4 | 7.00-7.42 | 29 | 3/3 | +0.25 | 0 | 1.00 | 0.00 |
| miller_group_fall_05_stair_misstep_night | 4 | 5.75-6.50 | 29 | 3/3 | +0.25 | 0 | 1.00 | 0.00 |
| miller_group_fall_08_fall_from_low_platform | 2 | 4.67-5.25 | 29 | 3/3 | +0.11 | 0 | 1.00 | 0.00 |
| miller_group_fall_10_slip_carrying_toolbox_night_rain | 4 | 4.17-5.08 | 29 | 3/3 | -0.14 | 0 | 1.00 | 0.00 |
| miller_night_fall_01_nv_slip_wet_grating | 4 | 5.42-6.08 | 2 | 0/3 | - | 0 | 0.65 | 0.00 |
| miller_night_fall_02_nv_trip_over_hose | 4 | 4.25-4.83 | 0 | 0/3 | - | 0 | 0.61 | 0.64 |
| miller_night_fall_05_nv_stumble_carrying_pipe | 5 | 4.92-5.42 | 9 | 2/3 | +0.21 | 0 | 1.00 | 0.00 |
| miller_night_fall_06_oh_slip_oily_patch | 2 | 4.00-4.50 | 30 | 3/3 | -0.03 | 0 | 1.00 | 0.00 |
| miller_night_fall_07_oh_trip_over_flange | 2 | 4.08-4.50 | 28 | 0/3 | - | 0 | 1.00 | 0.07 |
| miller_night_fall_08_oh_fall_from_low_platform | 3 | 4.92-5.50 | 28 | 3/3 | +0.47 | 0 | 1.00 | 0.00 |
| miller_night_fall_09_oh_slip_stepping_off_ladder | 3 | 5.08-5.58 | 30 | 3/3 | +0.22 | 0 | 1.00 | 0.00 |
| miller_night_fall_10_oh_slip_carrying_toolbox_alone | 1 | 4.42-4.83 | 24 | 2/3 | +0.71 | 0 | 1.00 | 0.00 |
| normal_t2_16_lie_under_pipe_rack | 3 | - | - | - | - | 0 | - | 0.84 |
| normal_t2_18_lie_on_side_drain_valve | 2 | - | - | - | - | 2 | - | 0.79 |
| normal_t2_19_get_down_and_slide_under | 3 | - | - | - | - | 0 | - | 0.04 |
| normal_t2_20_slide_out_and_stand | 3 | - | - | - | - | 0 | - | 0.42 |
| normal_t2_21_lie_prone_flashlight | 2 | - | - | - | - | 0 | - | 0.01 |
| normal_t2_22_kneel_one_knee_valve | 2 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_23_kneel_both_knees_flange | 3 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_25_kneel_check_gauge_rain | 1 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_27_kneel_sit_back_on_heels | 4 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_28_crouch_pump_gauge | 3 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_29_squat_pick_up_tools | 3 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_30_crouch_walk_under_low_pipe | 4 | - | - | - | - | 3 | - | 1.00 |
| normal_t2_31_crouch_group_talk_night | 5 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_32_deep_squat_bucket | 2 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_33_fast_drop_to_knee | 3 | - | - | - | - | 1 | - | 1.00 |
| normal_t2_34_fast_sit_on_toolbox | 3 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_35_duck_fast_swinging_load | 4 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_36_quick_down_to_look_under | 5 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_37_jump_down_squat_landing | 1 | - | - | - | - | 3 | - | 1.00 |
| normal_t2_39_sit_on_stairs | 1 | - | - | - | - | 0 | - | 0.08 |
| normal_t2_40_sit_then_lie_back_rest | 3 | - | - | - | - | 1 | - | 0.67 |
| normal_t2_41_slip_catch_handrail | 2 | - | - | - | - | 0 | - | 0.03 |
| normal_t2_42_trip_and_recover | 3 | - | - | - | - | 0 | - | 0.12 |
| normal_t2_43_slip_coworker_catches_arm | 5 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_44_stumble_hand_on_pipe | 3 | - | - | - | - | 0 | - | 0.24 |
| normal_t2_45_climb_down_ladder | 1 | - | - | - | - | 0 | - | 0.32 |
| normal_t2_46_walk_down_stairs_group | 1 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_47_step_down_off_platform | 4 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_48_walk_carry_group | 3 | - | - | - | - | 1 | - | 0.67 |
| normal_t2_49_turn_valve_wheel | 2 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_51_carry_hose_bottom_edge | 3 | - | - | - | - | 0 | - | 0.01 |
| normal_t2_52_carry_pipe_out_side_edge | 5 | - | - | - | - | 0 | - | 0.72 |
| normal_t2_53_crouch_at_frame_edge | 3 | - | - | - | - | 0 | - | 0.66 |
| normal_t2_54_quick_crouch_at_toolbox | 1 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_55_fast_kneel_at_hatch | 3 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_56_quick_squat_grab_radio | 2 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_57_fast_kneel_at_pipes_pair | 5 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_58_drop_to_hands_knees_leak | 3 | - | - | - | - | 0 | - | 0.01 |
| normal_t2_59_bend_into_hatch | 3 | - | - | - | - | 0 | - | 0.00 |
| normal_t2_60_lift_heavy_bag | 2 | - | - | - | - | 0 | - | 0.01 |
| normal_t2_61_crawl_under_low_pipe | 4 | - | - | - | - | 2 | - | 0.99 |
| normal_t2_62_get_up_fast_from_lying | 5 | - | - | - | - | 0 | - | 0.18 |
| normal_t2_65_crouch_hidden_behind_pump | 1 | - | - | - | - | 0 | - | 0.00 |
| normal_work_1p_1 | 3 | - | - | - | - | 0 | - | 0.33 |
| normal_work_1p_2 | 3 | - | - | - | - | 1 | - | 0.68 |
| normal_work_1p_4 | 3 | - | - | - | - | 0 | - | 0.34 |
| normal_work_1p_5 | 3 | - | - | - | - | 3 | - | 1.00 |
| normal_work_2p_1 | 5 | - | - | - | - | 0 | - | 0.02 |
| normal_work_2p_2 | 5 | - | - | - | - | 0 | - | 0.00 |
| normal_work_2p_3 | 5 | - | - | - | - | 0 | - | 0.00 |
| normal_work_2p_4 | 5 | - | - | - | - | 0 | - | 0.21 |
| normal_work_2p_5 | 5 | - | - | - | - | 2 | - | 1.00 |
| normal_work_2p_6 | 5 | - | - | - | - | 0 | - | 0.97 |
| normal_work_3p_1 | 2 | - | - | - | - | 0 | - | 0.02 |
| normal_work_3p_10 | 4 | - | - | - | - | 0 | - | 0.09 |
| normal_work_3p_2 | 2 | - | - | - | - | 0 | - | 0.28 |
| normal_work_3p_3 | 2 | - | - | - | - | 0 | - | 0.33 |
| normal_work_3p_4 | 2 | - | - | - | - | 0 | - | 0.39 |
| normal_work_3p_6 | 4 | - | - | - | - | 0 | - | 0.45 |
| normal_work_3p_7 | 4 | - | - | - | - | 2 | - | 1.00 |
| normal_work_3p_8 | 4 | - | - | - | - | 0 | - | 0.03 |
| normal_work_3p_9 | 4 | - | - | - | - | 0 | - | 0.00 |
| normal_work_4p_1 | 4 | - | - | - | - | 0 | - | 0.88 |
| oO8L7gEH7O6hbAxYdXVHD_output | 4 | 2.08-2.50 | 28 | 3/3 | +0.25 | 0 | 1.00 | 0.00 |
| slip_behind_person | 2 | 7.25-7.75 | 27 | 2/3 | +0.92 | 0 | 0.68 | 0.01 |
| slip_between_coworkers | 1 | 6.83-7.67 | 29 | 2/3 | +0.62 | 0 | 1.00 | 0.01 |
| slip_in_the_open | 5 | 6.33-6.67 | 30 | 2/3 | +0.83 | 0 | 1.00 | 0.03 |
| sudden_collapse | 3 | 6.25-6.67 | 29 | 3/3 | +0.00 | 1 | 1.00 | 0.65 |
| yQ3kWNc-Qe3EgqRaNso6l_alligator | 3 | - | - | - | - | 0 | - | 0.33 |

## Notes

- 2 fall clip(s) give no fall windows because the pose detector loses the faller mid-fall: fall_t2_10_backward_trip_toolbox_night, miller_night_fall_02_nv_trip_over_hose. They can only be caught if the model fires on the onset.

## Plots

![curves](plots/curves.png)

![precision_recall](plots/precision_recall.png)

![thresholds](plots/thresholds.png)

