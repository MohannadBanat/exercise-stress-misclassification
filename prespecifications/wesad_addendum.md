# External test on WESAD: addendum to m6_prespec.md (written before the run)

Written 2026-10-10, after data preparation and before any feature, model or outcome was computed on WESAD.
`m6_prespec.md` says that any WESAD-specific choice forced by its format is documented before the single run. This
document records those choices. Its SHA-256 is in `improve/wesad_addendum.sha256`. The run script refuses to run
twice: it writes `improve/WESAD_OPENED`.

## Data (prepared without any model or outcome)

**Participants.** The 15 WESAD participants as distributed (S2–S17; S1 and S12 are absent from the release).

**Signals.** The raw Empatica E4 wrist files (EDA, BVP, TEMP, ACC, HR) for each participant, from S#_E4_Data.zip.

**Labels.** The WESAD label stream (700 Hz, synchronised with the wrist signals in S#.pkl).
- It was aligned to the raw E4 clock by exact matching of the pkl wrist EDA samples against the raw EDA.csv.
- All 15 participants had a single matching offset, and 100% of samples matched.
- Blocks are runs of the same label lasting at least 60 s: 1 = baseline, 2 = Trier Social Stress Test (TSST),
  3 = amusement, 4 = meditation (two runs: medi1, medi2).
- Each participant has all five blocks, in the documented counterbalanced order.

## Processing (identical to training unless stated)

**Windows and features.**
- Windows: 30 s with a 15 s step, inside blocks only.
- The 29 features come from the unchanged functions of `pipeline/features.py` (the code is executed, not
  re-implemented).

**Calibrated features.** The difference from the mean of the participant's WESAD baseline windows.

This is the one forced difference from training: the WESAD baseline lasts about 20 min, against 3 min in
training.

**Context and adaptive features.**
- Context deltas (120 s) and 5-min adaptive differences are computed exactly as in training, over the same
  participant's earlier labelled windows.
- Causal smoothing over 45 s.

**Episodes and rest.**
- The stress episode is the TSST block, one per participant.
- The rest strata are baseline, amusement, medi1 and medi2.
- Rest windows within 60 s after the end of the TSST are excluded (grace period).
- Strict detection means an alarm event begins within [TSST start, TSST end + 60 s].
- Events are merged when less than 120 s apart, as in training.

## Models (fixed before the run)

**Selected detector.**
- The M6 rule (minimum P1 subject to the P2 and P3 constraints) is applied to grouped 5-fold out-of-fold predictions
  on all 36 training participants (100 null shifts).
- The selected candidate is refitted on all 36.
- τ is the 0.80 window-sensitivity quantile of the all-36 out-of-fold smoothed stress windows.

**Reference detector (added here, before the run, as a comparator).** The calibrated + context features with the
state alarm rule, refitted and thresholded in the same way.

## Endpoints

**Primary, for the selected detector:**
- **(i) Rest time in alarm (P1):** baseline, amusement and meditation windows outside the grace period, pooled.
- **(ii) Strict TSST detection:** the number of participants whose TSST was detected strictly, compared with chance.
  Chance is the mean over 1000 random circular shifts of each participant's score sequence. Excess = observed −
  expected, divided by 15.
- **(iii) Within-session area under the receiver operating characteristic curve (AUROC):** TSST against all rest
  windows, averaged over participants.

**Secondary:**
- alarm fraction by condition (baseline, amusement, meditation);
- TSST windows in alarm;
- any-overlap detection against chance;
- false alarms per hour at rest;
- paired differences, selected − reference, with participant-level bootstrap 95% CIs (10,000 resamples, seed 20261009).

**Statement of replication (fixed now).**
- The rest false-alarm improvement is called **replicated** if the paired difference in P1 (selected − reference)
  has a 95% CI wholly below 0 **and** the selected detector's strict excess over chance has a lower CI bound above 0.
- If only one of these holds, the result is **partly replicated**.
- If neither holds, it is **not replicated**.
- The result is reported whichever way it falls.

## Known differences that limit interpretation

- The stressor is social-evaluative (TSST, about 10 min).
- All conditions are seated.
- There is no exercise.
- The population is different.
- Amusement may itself raise arousal.
- With 15 participants, intervals will be wide.
