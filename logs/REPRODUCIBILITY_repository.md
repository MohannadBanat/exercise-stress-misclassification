# Reproducibility check of this repository

- **Date:** 14 October 2026, 17:32–21:28 UTC, Linux x86_64, Python 3.13.16, two CPU cores, packages as in `requirements.txt`.
- **What was run:** `code/run_all.sh` on a fresh copy of this repository, with the raw PhysioNet files (`HONGN_DIR`) and the WESAD folder prepared on 9 October 2026 by `code/wesad/prep_wesad.py` (`WESAD_E4_DIR`; the preparation itself was not re-run, its log is `logs/run_logs/wesad_prep_log.txt`).
- **Result:** all 46 archived output files compared by `code/verify.py` are identical (numbers to 1e-9), including the segment index, the sealed hold-out draw, the trial log, every registered, exploratory, detector, episode, false-alarm, WESAD and control result, and all 825 values in `results/vals.json`.

## Two fixes made during the check

1. Stage `19_WESAD_feat` first failed because `run_all.sh` did not create the `features_wesad/` output folder. The folder is now created at the start; the run was resumed at stage 19.
2. Stage `24_summarize` first failed because the post hoc response magnitudes (Section 4.10) had been computed inline on 9 October 2026 and appended to `sum_WESAD_diagnostics.json`, so no script produced them. `code/improve/wesad_magnitude.py` reproduces that computation (median over participants of each participant's median calibrated feature) and matches all 20 archived values; it is now stage `21b` of `run_all.sh`. The run was resumed at stage 24.

Neither fix changes any analysis code or result.

## Stage log

```
17:32:13 START 00_sha256
17:32:26 OK    00_sha256
17:32:26 START 01_inventory
17:32:34 OK    01_inventory
17:32:34 START 02_segments
17:32:34 OK    02_segments
17:32:34 START 03_timing
17:32:35 OK    03_timing
17:32:35 START 04_features
17:32:52 OK    04_features
17:32:52 START 05_features_v2
17:35:18 OK    05_features_v2
17:35:18 START 06_null_E1
17:52:53 OK    06_null_E1
17:52:53 START 07_registered
20:04:38 OK    07_registered
20:04:38 START 08_exploratory
20:33:52 OK    08_exploratory
20:33:52 START 09_seal_lockbox
20:33:52 OK    09_seal_lockbox
20:33:52 START 10_M1
20:38:46 OK    10_M1
20:38:46 START 11_M2
20:48:45 OK    11_M2
20:48:45 START 12_M3
20:55:16 OK    12_M3
20:55:16 START 13_M4_sealed
20:55:53 OK    13_M4_sealed
20:55:53 START 14_M5
20:58:58 OK    14_M5
20:58:58 START 15_M5_summary
20:58:59 OK    15_M5_summary
20:58:59 START 16_M6
21:13:08 OK    16_M6
21:13:08 START 17_M6_summary
21:13:10 OK    17_M6_summary
21:13:17 START 19_WESAD_feat
21:13:23 FAIL  19_WESAD_feat (see logs/19_WESAD_feat.log)
21:13:38 START 19_WESAD_feat
21:13:44 OK    19_WESAD_feat
21:13:44 START 20_WESAD_test
21:14:30 OK    20_WESAD_test
21:14:30 START 21_WESAD_diag
21:15:02 OK    21_WESAD_diag
21:15:02 START 22_controls1
21:22:20 OK    22_controls1
21:22:20 START 23_controls2
21:26:15 OK    23_controls2
21:26:15 START 24_summarize
21:26:15 FAIL  24_summarize (see logs/24_summarize.log)
21:27:35 OK    21b_WESAD_magnitude (run separately after the stage was added to run_all.sh)
21:27:35 START 24_summarize
21:27:36 OK    24_summarize
21:27:36 START 25_figures
21:27:43 OK    25_figures
21:27:43 START 26_verify
21:27:44 OK    26_verify
21:27:44 ALL DONE
```

## verify.py output

```
IDENTICAL        segment_index/segments.csv
IDENTICAL        segment_index/session_times.csv
IDENTICAL        segment_index/concurrent_groups.csv
IDENTICAL        segment_index/same_day_pairs.csv
IDENTICAL        segment_index/tag_qc.csv
IDENTICAL        segment_index/inventory.csv
IDENTICAL        prespecifications/lockbox.json
IDENTICAL        logs/trial_log.csv
IDENTICAL        results/vals.json
IDENTICAL        results/registered/E1_runs.csv
IDENTICAL        results/registered/E2_aer_anaer_lr_perparticipant.csv
IDENTICAL        results/registered/E2_aer_anaer_xgb_perparticipant.csv
IDENTICAL        results/registered/E2_four_lr_perparticipant.csv
IDENTICAL        results/registered/E2_four_xgb_perparticipant.csv
IDENTICAL        results/registered/E2_stress_rest_lr_perparticipant.csv
IDENTICAL        results/registered/E2_stress_rest_xgb_perparticipant.csv
IDENTICAL        results/registered/E3_four_perparticipant.csv
IDENTICAL        results/registered/E3_stress_rest_time_perparticipant.csv
IDENTICAL        results/registered/E5_perparticipant.csv
IDENTICAL        results/registered/E6_perparticipant.csv
IDENTICAL        results/registered/results.json
IDENTICAL        results/exploratory/X_deploy_perparticipant.csv
IDENTICAL        results/exploratory/X_four_perparticipant.csv
IDENTICAL        results/exploratory/X_stress_rest_perparticipant.csv
IDENTICAL        results/exploratory/exploratory.json
IDENTICAL        results/detector/res_M1.csv
IDENTICAL        results/detector/res_M2.csv
IDENTICAL        results/detector/res_M3_reference.csv
IDENTICAL        results/detector/res_M3_selected.csv
IDENTICAL        results/detector/res_M4_reference.csv
IDENTICAL        results/detector/res_M4_selected.csv
IDENTICAL        results/detector/res_M5_reference.csv
IDENTICAL        results/detector/res_M5_selected.csv
IDENTICAL        results/detector/res_WESAD_reference.csv
IDENTICAL        results/detector/res_WESAD_selected.csv
IDENTICAL        results/detector/sum_M1.json
IDENTICAL        results/detector/sum_M2.json
IDENTICAL        results/detector/sum_M3.json
IDENTICAL        results/detector/sum_M4.json
IDENTICAL        results/detector/sum_M5.json
IDENTICAL        results/detector/sum_M6.json
IDENTICAL        results/detector/sum_WESAD.json
IDENTICAL        results/detector/sum_WESAD_diagnostics.json
IDENTICAL        results/controls/sum_review.json
IDENTICAL        segment_index/segments_wesad.csv
IDENTICAL        segment_index/sync_wesad.csv

46 files compared, 0 not identical; manuscript values archived: 825
```
