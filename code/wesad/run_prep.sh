#!/bin/bash
# Extract the raw E4 files of each WESAD participant and align the WESAD labels to the raw E4 clock (prep_wesad.py),
# one participant per call as in the original run, then combine the per-participant files.
# Usage: bash wesad/run_prep.sh /path/to/WESAD.zip <out_dir>
set -e; ZIP="$1"; OUT="$2"; HERE="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$OUT"
for s in S2 S3 S4 S5 S6 S7 S8 S9 S10 S11 S13 S14 S15 S16 S17; do python3 "$HERE/prep_wesad.py" "$ZIP" "$OUT" $s; done
python3 - "$OUT" <<'PY'
import sys, glob, os, pandas as pd
o = sys.argv[1]
for k in ('segments', 'sync'):
    fs = sorted(glob.glob(os.path.join(o, f'{k}_wesad_S*.csv')))
    pd.concat([pd.read_csv(f) for f in fs], ignore_index=True).to_csv(os.path.join(o, f'{k}_wesad.csv'), index=False)
    print(k, len(fs), 'participants')
PY
