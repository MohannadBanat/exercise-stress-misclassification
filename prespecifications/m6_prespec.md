# M6: reducing false alarms at rest. Pre-specification

Written 2026-10-10, before any M6 number was computed. The SHA-256 of this file is recorded in
`improve/m6_prespec.sha256`, and a copy goes to the authors' computer and the project folder. Anything decided
after results are seen is reported as post hoc.

## Status and motivation

This analysis is exploratory. The sealed hold-out is spent, and 100 earlier configurations were evaluated on
these 36 participants.

The final detector flags about half of the rest windows, mostly early in post-task rest and in the opening
baseline. When scored per episode, its alarm is often already on before a task begins. A plausible mechanism is
that the features follow slowly varying arousal (tonic level) that outlasts each task. Both interventions below
target that mechanism and nothing else.

## Fixed (not tuned)

- Model: extremely randomized trees.
- Causal smoothing: 45 s.
- Threshold τ: the 0.80 window-sensitivity quantile of the training participants' out-of-fold smoothed stress
  windows, computed separately for each feature set.
- Episode definitions, consecutive-window rule, 120 s event merging and 60 s grace period: as in
  `episode_prespec.md` (M5).
- All 36 participants are evaluated with nested leave-one-participant-out (LOPO).

## Candidates (4 feature sets × 3 alarm logics = 12)

### Feature sets

Let PHYS be the 19 physiological features. Let DEL be the 8 existing 120 s context deltas. Let ACC be the 10
accelerometer features.

- **ref**: cal(PHYS) + DEL + ACC. This is the current detector. cal is the difference from the participant's
  stress-session baseline mean.
- **ad5**: ad5(PHYS) + DEL + ACC.
  - ad_W(x) at window t = x_t − the median of x over windows of the same session that start in
    [t − W, t − 30 s].
  - These are earlier, non-overlapping windows only. At least 4 such windows are required, otherwise the value
    is missing (median-imputed by the model pipeline). W = 5 min.
- **ad10**: ad10(PHYS) + DEL + ACC (W = 10 min).
- **refad5**: cal(PHYS) + DEL + ad5(PHYS) + ACC.

The trailing medians use all windows of the session in `features/windows.csv`, labelled or not as stress/rest.
They use no labels.

### Alarm logics

- **state**: a window is in alarm when its smoothed score s ≥ τ. This is the current rule, the same as M5 with
  N = 1.
- **hold2 / hold4**: an alarm begins at a window where s ≥ τ and the detector is armed. It stays on for H =
  2 or 4 min of decision time and then ends, whatever the score.
  - The detector starts armed.
  - It is disarmed at each alarm onset.
  - It is re-armed whenever s < τ.
  - So a new alarm needs the score to have dropped below τ since the previous onset.
  - Alarm-state windows form events exactly as in M5.

## Endpoints

1. **P1 (primary):** rest time in alarm. This is the fraction of rest-labelled windows of the stress session
   (baseline included) that are in the alarm state, excluding windows in the 60 s grace period after each
   episode. It is pooled over participants.
2. **P2 (detection):** strict episode detection in excess of chance.
   - Strict detection: an alarm event begins within [episode start, episode end + 60 s] (M5 strict scoring).
   - Chance: the mean strict detection over random circular shifts of each participant's stress-session score
     sequence, using 1000 shifts in outer evaluation and 100 in inner selection.
   - Excess = (detected − expected by chance) / episodes, pooled.
3. **P3 (guard):** exercise time in alarm, the fraction of exercise-session windows in the alarm state, pooled.
4. **Secondary:**
   - rest time in alarm excluding the baseline;
   - baseline time in alarm;
   - any-overlap episode detection and its chance level;
   - false alarms per hour at rest, at baseline and during exercise (M5 definitions);
   - fraction of stress windows in alarm;
   - within-session area under the receiver operating characteristic curve (AUROC) for each feature set
     (threshold-free).

## Selection inside each training fold (nested LOPO)

- Inner out-of-fold scores come from grouped 5-fold cross-validation on the 35 training participants.
- **Eligible candidates** are those whose inner P2 ≥ the reference's inner P2 − 0.05 **and** whose inner P3 ≤
  the reference's inner P3 + 0.02.
- **Choice:** the eligible candidate with the lowest inner P1.
  - The reference (ref + state) is always eligible.
  - Ties go to the earlier candidate in this order: ref-state, ref-hold2, ref-hold4, ad5-…, ad10-…, refad5-….

## Comparison and allowed claim

The candidates chosen inside the folds are compared with the reference (ref + state) using paired
participant-level bootstrap CIs (10,000 resamples, seed 20261009) for the pooled differences.

An improvement is claimed only if all three hold:
- **(a)** the 95% CI of the P1 difference lies wholly below 0;
- **(b)** the lower CI bound of the P2 difference is above −0.05;
- **(c)** the upper CI bound of the P3 difference is below 0.05.

Otherwise the result is reported as no clear gain. Outer results for all 12 fixed candidates are reported as
descriptive only.

## External test (only if WESAD becomes available)

- **Final configuration:** selected by the same rule using grouped 5-fold CV on all 36 participants, then
  refitted on all 36.
- **Applied once** to the Empatica E4 wrist data of WESAD:
  - 30 s windows with a 15 s step and the same features;
  - calibration against each participant's WESAD baseline condition;
  - stress episode: the Trier Social Stress Test (TSST);
  - rest strata: baseline, amusement and meditation.
- **Endpoints:**
  - P1 on the rest strata;
  - strict detection of the TSST episode against chance;
  - within-session AUROC.
- **Adaptation:** any WESAD-specific choice forced by its format is documented before the single run.

## Multiplicity

All 12 candidates are appended to `improve/trial_log.csv` with tag `M6_rest`.
