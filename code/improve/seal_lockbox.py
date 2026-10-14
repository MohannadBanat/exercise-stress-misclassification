"""Seal a stratified, random hold-out ("lockbox") of 12 participants for the improved-detector work.
Strata: 6 of the 18 V1 participants, 4 of the 13 V2 participants with exercise sessions, 2 of the 5 V2 stress-only
participants. The list and its SHA-256 are written once; development code must never load these participants."""
import json, hashlib, numpy as np, os
OUT = 'improve/lockbox.json'
if os.path.exists(OUT): raise SystemExit('lockbox already sealed; refusing to redraw')
rng = np.random.default_rng(20261009)
v1 = [f'S{i:02d}' for i in range(1, 19)]; v2ex = [f'f{i:02d}' for i in range(1, 14)]; v2s = [f'f{i:02d}' for i in range(14, 19)]
box = sorted(list(rng.choice(v1, 6, replace=False)) + list(rng.choice(v2ex, 4, replace=False)) + list(rng.choice(v2s, 2, replace=False)))
payload = json.dumps(box)
rec = dict(lockbox=box, sha256=hashlib.sha256(payload.encode()).hexdigest(), seed=20261009, sealed='2026-10-09',
           rule='6 of 18 V1, 4 of 13 V2 with exercise, 2 of 5 V2 stress-only; drawn once with the seed above')
json.dump(rec, open(OUT, 'w'), indent=1); print(json.dumps(rec, indent=1))
