# Changelog

## 1.1.1 (12 November 2026): second revision for *Sensors*

At a reviewer's request, the file-integrity details were removed from the manuscript and are now documented in this repository. No analysis code, data file or result was changed; this version adds:

- `README.md`: a new section, "Data and file integrity", which documents the SHA-256 check of the 708 files of the PhysioNet release, the full hashes and recording times of the three pre-specification documents, and the hash of the sealed-participant list. The previous README table pointed to Table A2 of the first revision, which no longer lists the hashes.
- `code/make_figure1_v12.py`: Figure 1 of the second revision, which labels the exploratory analyses EA1–EA5 and adds the post hoc controls PC1–PC7.
- `CITATION.cff` and `.zenodo.json`: version 1.1.1.

## 1.1.0 (31 October 2026): revision for *Sensors*

The manuscript was revised after peer review and retitled "Revealing Misclassification of Exercise Sessions in Wearable Stress Detection for Automated Insulin Delivery". Nothing archived in v1.0.0 was changed; this version adds:

- `code/improve/review_r1_ablations.py`: post hoc ablations of the reference detector (physiological and accelerometer features, physiological features only, accelerometer features only) under grouped five-fold cross-validation on all 36 participants, in the sealed test (the full-feature row reproduces the sealed test) and on WESAD. Output: `results/controls/sum_r1_ablations.json`; log: `logs/run_logs/review_R1_ablations.txt`.
- `code/improve/review_r1_magnitude.py`: stress-window detection by the rise in pulse rate and tonic EDA from baseline, in bins shared by both datasets, for the reference detector and the detector with 5-min adaptive differences, under cross-validation and on WESAD. Output: `results/controls/sum_r1_magnitude.json`; log: `logs/run_logs/review_R1_magnitude.txt`.
- `code/make_vals_revision.py` and `results/vals_revision.json`: every number these analyses add to the paper (Section 4.11, Tables 11 and 12).
- `code/make_revision_figures.py`: the redrawn Figure 1 and the new Figures 3 and 8 of the revised manuscript.
- `code/run_all.sh` and `code/verify.py` run and check the new analyses; `README.md`, `CITATION.cff` and `.zenodo.json` describe this version.

## 1.0.0 (17 October 2026)

Code, segment index, pre-specified rules, trial log, logs, archived results and pinned environment of the original submission.
