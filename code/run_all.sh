#!/bin/bash
# End-to-end rerun of every analysis in the paper, from the raw data to vals.json (every number in the manuscript).
# Usage (from the repository root):
#   HONGN_DIR=/path/to/wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1 \
#   WESAD_ZIP=/path/to/WESAD.zip  bash code/run_all.sh
# WESAD_ZIP is optional (or WESAD_E4_DIR, a folder already prepared by code/wesad/run_prep.sh); without either, the external test is skipped. The run takes about 5 h on two CPU cores.
# All outputs go to WORK_DIR (default: ./work); the repository itself is not modified. Stops at the first failing stage.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; DS="$(cd "${HONGN_DIR:?set HONGN_DIR to the PhysioNet dataset folder}" && pwd)"
WORK="${WORK_DIR:-$ROOT/work}"; mkdir -p "$WORK"; cp -r "$ROOT/code/." "$WORK/"; cd "$WORK"; mkdir -p logs data features features_wesad build fig
stage() { name=$1; shift; echo "$(date -u +%H:%M:%S) START $name" >> logs/status.txt
  if "$@" > logs/$name.log 2>&1; then echo "$(date -u +%H:%M:%S) OK    $name" >> logs/status.txt; else echo "$(date -u +%H:%M:%S) FAIL  $name (see logs/$name.log)" >> logs/status.txt; exit 1; fi; }
stage 00_sha256        bash -c "cd '$DS' && sha256sum -c SHA256SUMS.txt --ignore-missing --quiet"
stage 01_inventory     python -I analysis/01_inventory.py "$DS" data/inventory.csv
stage 02_segments      python -I analysis/02_segment_index.py "$DS" data
stage 03_timing        python -I analysis/03_session_timing.py "$DS" data
stage 04_features      python -I pipeline/features.py "$DS" data/segments.csv data/session_times.csv data/concurrent_groups.csv features
stage 05_features_v2   python -I improve/features_v2.py "$DS" features/windows.csv features/windows_v2.csv
stage 06_null_E1       python -I pipeline/experiments.py features results_null_e1 --only E1
stage 07_registered    python -I pipeline/experiments.py features results_real --osf https://osf.io/swge8
stage 08_exploratory   python -I pipeline/exploratory.py features results_x --osf https://osf.io/swge8
stage 09_seal_lockbox  python improve/seal_lockbox.py
stage 10_M1            python improve/run_m1.py
stage 11_M2            python improve/run_m2.py
stage 12_M3            python improve/run_m3.py
cp "$ROOT/prespecifications/"*.md "$ROOT/prespecifications/"*.sha256 improve/
stage 13_M4_sealed     python improve/run_m4.py --i-understand-this-runs-once
stage 14_M5            python improve/run_m5_episode.py
stage 15_M5_summary    python improve/summarize_m5.py
stage 16_M6            python improve/run_m6.py
stage 17_M6_summary    python improve/summarize_m6.py
if [ -n "${WESAD_ZIP:-}" ] || [ -n "${WESAD_E4_DIR:-}" ]; then
  if [ -n "${WESAD_E4_DIR:-}" ]; then cp -r "$WESAD_E4_DIR" wesad_e4   # already extracted and aligned by wesad/run_prep.sh
  else stage 18_WESAD_prep  bash wesad/run_prep.sh "$WESAD_ZIP" wesad_e4; fi
  cp wesad_e4/segments_wesad.csv data/segments_wesad.csv
  stage 19_WESAD_feat  python -I improve/wesad_features.py wesad_e4 features_wesad/windows_wesad.csv
  stage 20_WESAD_test  python improve/run_wesad.py --i-understand-this-runs-once
  stage 21_WESAD_diag  python improve/wesad_diagnostics.py
  stage 21b_WESAD_magnitude python improve/wesad_magnitude.py
fi
stage 22_controls1     python improve/review_controls.py S1 S2 S3
stage 23_controls2     python improve/review_controls2.py S4 S5 S6 S7 S4b
stage 24_summarize     python summarize.py
stage 25_figures       bash -c "python make_figures.py && python make_result_figures.py fig3 fig4 fig5 fig6 fig7 fig8"
stage 26_verify        python verify.py "$ROOT"
echo "$(date -u +%H:%M:%S) ALL DONE" >> logs/status.txt; cat logs/26_verify.log
