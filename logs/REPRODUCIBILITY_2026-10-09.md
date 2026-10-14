# Reproducibility check: full rerun from the raw data

- **Date:** 2026-10-09, 17:59–22:11 UTC.
- **Driver:** `run_all.sh` (21 stages, sequential, two-core CPU).
- **Method:** a fresh directory with only the code, the raw PhysioNet files and the episode pre-specification; every result file then compared with the original run by `compare.py`.

## Outcome

- **All 21 stages completed.**
- **Every result file is identical to the original** (numeric tolerance 1e-9), apart from one input file noted below:
  - the registered results (`results.json`, all 10 confirmatory tests, E1 600 runs);
  - the shuffled-label control;
  - the exploratory analysis;
  - the hold-out draw (same 12 participants, same SHA-256);
  - detector development M1–M3;
  - the sealed-test computation M4 (re-executed here only as a reproducibility check, in a separate folder; selection is done on the development participants, so no new information was used);
  - the episode evaluation M5;
  - all post hoc controls.
- **All 583 numbers in the manuscript (`vals.json`) are identical.**
- **Checksums:** 704 data files verified. The four documentation files of the release (LICENSE, README, notebook, data constraints) were not in the workspace copy; they were verified earlier on the original download.
- **`windows_v2.csv`:** differs by at most 9e-7 (in 9 EDA-decomposition columns). This is the numerical tolerance of the convex cvxEDA solver. No downstream result changed.

## Code fixes made during the rerun

1. `improve/framework.py` and `improve/framework2.py` now write the trial log with the same five columns as the final log (tag, candidate, mean_inner_score, time, score). The original log's header had been renamed by hand; the values are identical.
2. The pre-task baseline comparison (S4b), first computed inline, is now part of `improve/review_controls2.py`.
3. Manuscript wording: '708 files of the release (704 data files in 100 recordings and four documentation files)'.

## Stage log

```
17:59:52 START 00_sha256
17:59:54 OK    00_sha256
17:59:54 START 01_inventory
18:00:00 OK    01_inventory
18:00:00 START 02_segments
18:00:00 OK    02_segments
18:00:00 START 03_timing
18:00:02 OK    03_timing
18:00:02 START 04_features
18:00:23 OK    04_features
18:00:23 START 05_features_v2
18:03:15 OK    05_features_v2
18:03:15 START 06_null_E1
18:24:38 OK    06_null_E1
18:24:38 START 07_registered
20:50:21 OK    07_registered
20:50:21 START 08_exploratory
21:26:13 OK    08_exploratory
21:26:13 START 09_seal_lockbox
21:26:13 OK    09_seal_lockbox
21:26:13 START 10_M1
21:31:55 OK    10_M1
21:31:55 START 11_M2
21:43:57 OK    11_M2
21:43:57 START 12_M3
21:52:03 OK    12_M3
21:52:03 START 13_M4_repro
21:52:50 OK    13_M4_repro
21:52:50 START 14_M5
21:56:49 OK    14_M5
21:56:49 START 15_M5_summary
21:56:50 OK    15_M5_summary
21:56:50 START 16_controls1
22:06:18 OK    16_controls1
22:06:18 START 17_controls2
22:11:05 OK    17_controls2
22:11:05 START 18_summarize
22:11:05 OK    18_summarize
22:11:05 START 19_figures
22:11:14 OK    19_figures
22:11:14 START 20_manuscript
22:11:15 OK    20_manuscript
22:11:15 ALL DONE
```

## Comparison output

```
IDENTICAL              data/segments.csv
IDENTICAL              data/tag_qc.csv
IDENTICAL              data/session_times.csv
IDENTICAL              data/same_day_pairs.csv
IDENTICAL              data/concurrent_groups.csv
IDENTICAL              data/inventory.csv
IDENTICAL              features/windows.csv
IDENTICAL              features/e1_blocks.csv
9 differences          features/windows_v2.csv
IDENTICAL              results_null_e1/E1_runs.csv
IDENTICAL              improve/lockbox.json
IDENTICAL              results_real/E5_perparticipant.csv
IDENTICAL              results_real/E6_perparticipant.csv
IDENTICAL              results_real/E2_stress_rest_xgb_perparticipant.csv
IDENTICAL              results_real/E2_aer_anaer_lr_perparticipant.csv
IDENTICAL              results_real/results.json
IDENTICAL              results_real/E2_four_xgb_perparticipant.csv
IDENTICAL              results_real/E1_runs.csv
IDENTICAL              results_real/E2_aer_anaer_xgb_perparticipant.csv
IDENTICAL              results_real/E2_four_lr_perparticipant.csv
IDENTICAL              results_real/E2_stress_rest_lr_perparticipant.csv
IDENTICAL              results_real/E3_stress_rest_time_perparticipant.csv
IDENTICAL              results_real/E3_four_perparticipant.csv
IDENTICAL              results_x/exploratory.json
IDENTICAL              results_x/X_deploy_perparticipant.csv
IDENTICAL              results_x/X_stress_rest_perparticipant.csv
IDENTICAL              results_x/X_four_perparticipant.csv
IDENTICAL              improve/sum_M1.json
IDENTICAL              improve/sum_M2.json
IDENTICAL              improve/sum_M3.json
IDENTICAL              improve/sum_M4.json
IDENTICAL              improve/sum_M5.json
IDENTICAL              improve/res_M1.csv
IDENTICAL              improve/res_M2.csv
IDENTICAL              improve/res_M3_selected.csv
IDENTICAL              improve/res_M3_reference.csv
IDENTICAL              improve/res_M4_selected.csv
IDENTICAL              improve/res_M4_reference.csv
IDENTICAL              improve/res_M5_selected.csv
IDENTICAL              improve/res_M5_reference.csv
IDENTICAL              improve/trial_log.csv
IDENTICAL              improve/review/sum_review.json
IDENTICAL              improve/review/S2_all.csv
IDENTICAL              improve/review/S3_e6_events.csv
IDENTICAL              improve/review/S2_phys_only.csv
IDENTICAL              improve/review/auroc_det_vs_time.csv
IDENTICAL              improve/review/S2_acc_only.csv

manuscript values: 583 in original, 583 in rerun, 0 differ
```
