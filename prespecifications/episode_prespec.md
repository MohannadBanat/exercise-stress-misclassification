# M5: episode-level evaluation of the stress detector. Pre-specification

Written 2026-10-09, before any episode-level number was computed. The SHA-256 of this file is recorded in
`improve/episode_prespec.sha256`. A copy is saved on the authors' computer and in the project folder at the same time.
Anything decided after results were seen will be labelled as post hoc in the manuscript.

## Status

Exploratory, not part of the OSF registration (https://osf.io/swge8). The sealed hold-out was opened in M4. Its
12 participants were also part of the earlier exploratory analysis. This analysis therefore uses all 36
participants in leave-one-participant-out (LOPO) evaluation, and every result is reported as exploratory.

## What is fixed and not tuned

- **Model pipeline.** The final detector from M4 (`REF`): calibrated + context features (`v1_cal_ctx`),
  extremely randomized trees, a single model, all training labels, causal smoothing over 45 s, global threshold
  rule. The model and features are not changed.
- **Windows.** 30 s windows with a 15 s step. A window's decision time is its end (`end_s`).

## Structural facts checked before writing (no outcomes looked at)

- Window counts are one or two per short task: real opinion, opposite opinion and subtract each last about
  30–40 s.
- Stroop lasts about 110 s and the TMCT 140–270 s.
- Scored windows of neighbouring blocks start at most 44 s apart, except for:
  - baseline → first task (107–212 s, instructions);
  - real → opposite opinion (48–104 s);
  - opposite opinion → subtract in V1 (53–80 s);
  - two exercise transitions.

## Definitions

1. **Stress episode.** A maximal run of stress blocks in a stress session that is not separated by a rest block.
   - V1: {Stroop}, {TMCT}, {real opinion, opposite opinion, subtract}.
   - V2: {TMCT}, {real opinion, opposite opinion}, {subtract}.
   - This gives three episodes per participant and 108 in total.
   - An episode's start is the start of its first block; its end is the end of its last block (from
     `segments.csv`).
2. **Consecutive windows.** Two successive scored windows of a session count as consecutive if their start times
   differ by at most 45 s, or if both lie inside the same stress episode. Otherwise the run counter resets.
3. **Alarm state.** At each window, `s` is the causally smoothed probability. The window is in the alarm state if
   `s ≥ τ` held for at least `N` consecutive windows up to and including it.
4. **Alarm events.**
   - An event is a maximal run of alarm-state windows. Two events of the same session merge if fewer than 120 s
     separate the last alarm decision of one from the first alarm decision of the next.
   - An event spans [first alarm decision time, last alarm decision time].
5. **Grace period.** `G = 60 s` after each episode end. The decision on a window can include up to 45 s of
   smoothing and 30 s of window from before that decision.
6. **Detected episode (any-overlap scoring, as in seizure-detection standards).** An episode is detected if any
   alarm event overlaps [episode start, episode end + G].
7. **False alarm.** An event that overlaps no [episode start, episode end + G] interval. It is counted in the
   stratum of its first alarm window:
   - **baseline:** the stress-session baseline;
   - **rest:** non-baseline rest outside grace periods;
   - **exercise:** all exercise-session windows.
8. **Exposure time.** The number of scored windows in a stratum (rest windows whose decision time falls inside a
   grace period are excluded) × 15 s.
9. **Rates.**
   - **Episode sensitivity:** detected episodes / all episodes, pooled. This equals the participant mean, since
     everyone has three episodes.
   - **False alarms per hour (FA/h):** pooled per stratum, as total false alarms / total exposure hours.
   - **Combined false-alarm rate:** FA/h_comb = mean(FA/h rest, FA/h exercise), mirroring the window-level
     specificity.
   - **CIs:** 95% intervals from a participant-level bootstrap (10,000 resamples, seed 20261009), recomputing the
     pooled ratios in each resample.
10. **Latency.** For each detected episode, the first alarm-state decision time within
    [episode start, episode end + G] minus episode start. It is 0 if an event was already active at episode start.

## Operating rule chosen inside training folds only

- **Threshold.** τ is the quantile of the training out-of-fold smoothed stress-window scores giving a
  window-level sensitivity target `q ∈ {0.5, 0.6, 0.7, 0.8, 0.9}`.
- **Persistence.** `N ∈ {1, 2, 3, 4, 6, 8}` windows. N = 8 means 105 s of sustained threshold crossing, about
  2 min of signal including the window. This is the example raised in planning.
- **Grid.** 30 rules.
- **Selection objective (on the training participants' grouped 5-fold out-of-fold predictions):**
  - minimise FA/h_comb among rules with pooled episode sensitivity ≥ 0.90;
  - if none qualifies, take the highest episode sensitivity;
  - break ties by lower FA/h_comb, then smaller N.
- **Outer evaluation.** For each held-out participant, refit on the other 35, apply the selected (q, N), and
  score.

## Comparison (pre-specified)

- **Reference.** The same pipeline at the window-level operating point already in the manuscript: q = 0.80,
  N = 1, scored with the same episode rules.
- **Endpoints (selected − reference, paired by participant, bootstrap CI):**
  1. Difference in FA/h_comb, plus FA/h rest and FA/h exercise separately.
  2. Difference in episode sensitivity, with a non-inferiority margin of −0.10. One missed episode is 0.33 for a
     participant, so 0.10 is about one extra miss in every three participants.
- **Allowed claim.** "The persistence rule reduces false alarms without losing episodes" is claimed only if the
  CI for the FA/h_comb difference lies wholly below 0 **and** the lower CI bound of the sensitivity difference is
  above −0.10. Otherwise the result is reported as no clear gain.

## Secondary and descriptive (pre-listed)

- Episode detection rate by episode type: Stroop; TMCT; opinion cluster (V1 opinion + subtract, V2 opinion
  pair); V2 subtract.
- Latency median and IQR.
- Baseline FA/h.
- Fraction of rest decisions (outside grace) in the alarm state.
- Mean of per-participant FA/h, as a check on pooling.
- **Sensitivity analyses, for the selected and reference rules:**
  - (a) G = 0;
  - (b) strict onset scoring, where an episode is detected only if an event *begins* in
    [start, end + G] and false alarms are events that begin outside every such interval.
- **Descriptive operating curve.** Pooled episode sensitivity vs FA/h_comb for all 30 grid rules on the outer
  LOPO predictions. It is labelled descriptive, because choosing from it would use test data.

## Multiplicity record

All 30 rules and the reference are appended to `improve/trial_log.csv` with tag `M5_episode`.
