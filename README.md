# exercise-stress-misclassification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23438148.svg)](https://doi.org/10.5281/zenodo.23438148)

Code, segment index and pre-specified analysis rules for:

> Abu Owida, H.; Alsarhan, A.; Banat, M.; Al-Fraihat, D.; Algarni, A. **Revealing Misclassification of Exercise Sessions in Wearable Stress Detection for Automated Insulin Delivery.** *Sensors* (under revision, 2026; submitted as "Revealing Exercise-Induced Misclassification in Wearable Stress Detection for Automated Insulin Delivery").

The study re-evaluates stress and exercise recognition from Empatica E4 wrist signals under a pre-registered plan, measures how often a stress detector fires during structured exercise, and tests an exercise-robust detector on a sealed test set and on an external dataset (WESAD).

## Archived version

Version 1.0.0, cited in the original submission, is archived on Zenodo: https://doi.org/10.5281/zenodo.23438149. Version 1.1.0 adds the analyses of the first revision, and version 1.1.1 documents the integrity checks below and adds the script for Figure 1 of the second revision (see `CHANGELOG.md`). The concept DOI https://doi.org/10.5281/zenodo.23438148, cited in the revised manuscript, always resolves to the latest version.

## Pre-registration

The analysis plan was registered on the Open Science Framework before any model was trained: https://osf.io/swge8

Rules for the analyses added after registration (episode-level evaluation, reduction of rest false alarms, external test) were fixed in dated documents before each analysis was run. Those documents are in `prespecifications/`; their SHA-256 hashes are listed under [Data and file integrity](#data-and-file-integrity).

## Data (not included)

The raw recordings are **not** redistributed here. Download them from the original sources:

| Dataset | Source |
|---|---|
| Wearable device dataset from induced stress and structured exercise sessions, v1.0.1 (Hongn et al.) | PhysioNet, https://doi.org/10.13026/he0v-tf17 |
| WESAD (Schmidt et al.) | UCI Machine Learning Repository |

## Repository contents

| Folder | Contents |
|---|---|
| `code/` | Analysis scripts: integrity checks and segment mapping (`analysis/`), windowing, features and the registered experiments E1–E6 (`pipeline/`), detector development, sealed test, episode-level evaluation, rest false alarms, external test and post hoc controls (`improve/`), WESAD preparation (`wesad/`), the number summary (`summarize.py`), figures, and the end-to-end driver `run_all.sh` with its check `verify.py` |
| `segment_index/` | Mapping of every event mark to a protocol block (`segments.csv`, 885 blocks) with the session timing, quality-control and grouping files derived from it; WESAD condition blocks aligned to the raw E4 clock (`segments_wesad.csv`, `sync_wesad.csv`) |
| `prespecifications/` | Hashed rule documents for the episode-level, false-alarm and external analyses, and the record of the sealed hold-out (`lockbox.json`) |
| `logs/` | Trial log of all configurations evaluated during detector development (`trial_log.csv`), the logs of the original runs, the dates on which the sealed hold-out and WESAD were first opened, and the record of an independent full rerun |
| `results/` | Archived outputs of the original run, and `vals.json`, every number quoted in the paper; since v1.1.0 also the outputs of the analyses added in the revision (`controls/sum_r1_ablations.json`, `controls/sum_r1_magnitude.json`) and `vals_revision.json`, every number they add to the paper |
| `environment/` | Complete `pip freeze`, Python version and platform of the environment that produced the results |

## Data and file integrity

Up to the first revision these checks were described in the manuscript; from the second revision on they are documented here.

**Dataset files.** The PhysioNet release (v1.0.1) publishes SHA-256 checksums of its files in `SHA256SUMS.txt`. All 708 files of the release (704 data files in 100 recordings and four documentation files: LICENSE, README, the notebook and the data-constraints file) were checked against them on the original download, and all matched. `code/run_all.sh` repeats the check as its first stage (`00_sha256`: `sha256sum -c SHA256SUMS.txt --ignore-missing --quiet`, run in the dataset folder) and stops if any file differs. In the full rerun of 9 October 2026 all 704 data files matched; the documentation files were not in that copy (`logs/REPRODUCIBILITY_2026-10-09.md`).

**Pre-specification documents.** The rules of the episode-level evaluation, the reduction of rest false alarms and the external test (EA3–EA5 in the second revision) were fixed in the documents below before the analysis each one governs was run.

| Document | SHA-256 | Recorded (UTC) |
|---|---|---|
| `episode_prespec.md` (episode-level evaluation) | `409b985cbfbbb000f66471ef3b6277177ca22461fb6041f8759b08d2eb9bea75` | 2026-10-09T01:34:33Z |
| `m6_prespec.md` (reduction of rest false alarms) | `2ef370e3330a82462c1da16d33ba402507d2bda10fbb9fcfd389723b8f60d657` | 2026-10-09T22:19:57Z |
| `wesad_addendum.md` (external test on WESAD) | `7d87c337fccf15589beead721e0d082e9a21480be5798e9cae8c3205f147babe` | 2026-10-09T23:40:07Z |

Each `.sha256` file is the original record: the hash on the first line and the time it was taken (UTC) on the second. Check them with `cd prespecifications && sha256sum *.md`. The scripts that use these rules (`improve/run_m5_episode.py`, `improve/run_m6.py`, `improve/run_wesad.py`) refuse to run if a document has changed. The hashes were not deposited with a third party, so the time at which each document was fixed cannot be verified independently.

**Sealed participants.** `prespecifications/lockbox.json` records the 12 sealed participants, the seed with which they were drawn, the date on which they were sealed (2026-10-09) and the SHA-256 hash of their list (`13999c34317c8dc1cf015c2aa16e0764557d9c741f3959d89fccf638dd7e8664`); `improve/run_m4.py` refuses to run if the list has been altered. `logs/run_once_markers.txt` records when the sealed participants and WESAD were first opened.

## Environment

Analyses were run in Python 3.13 with scikit-learn 1.9, XGBoost 3.4, statsmodels 0.15 and NeuroKit2 0.2.13 on two CPU cores (exact versions in `requirements.txt`, full list in `environment/pip-freeze.txt`). Seeds: 20261008 (registered analyses) and 20261009 (detector work). Figures use the TeX Gyre Pagella font if it is installed.

```bash
pip install -r requirements.txt
```

## Reproducing the results

```bash
HONGN_DIR=/path/to/wearable-device-dataset-from-induced-stress-and-structured-exercise-sessions-1.0.1 \
WESAD_ZIP=/path/to/WESAD.zip \
bash code/run_all.sh
```

The driver copies `code/` to `work/`, runs every stage from the raw files (about 5 h on two cores; without `WESAD_ZIP` the external test is skipped), and finishes with `verify.py`, which compares every output, and every number in the paper, with the archived files in `results/`, `segment_index/`, `prespecifications/` and `logs/`.

Scripts that open the sealed hold-out or WESAD (`improve/run_m4.py`, `improve/run_wesad.py`) refuse to run twice in the same directory and require an explicit flag; `logs/run_once_markers.txt` records when they were first run. Compared with the scripts as originally run, only one hard-coded file path (`improve/run_wesad.py`) and four usage comments were changed. One step had been run inline rather than from a script, the post hoc response magnitudes reported for WESAD; `improve/wesad_magnitude.py` was written on 14 October 2026 to reproduce it and matches all 20 archived values.

On 14 October 2026 the repository was run end to end from a fresh copy: every one of the 46 archived output files, including all 825 values in `results/vals.json`, was reproduced identically (`logs/REPRODUCIBILITY_repository.md`).

## Analyses added in the revision (v1.1.0)

At the reviewers' request, two post hoc analyses were added: ablations of the reference detector with physiological features only or accelerometer features only, under grouped cross-validation, in the sealed test and on WESAD (`improve/review_r1_ablations.py`), and stress-window detection by the size of the response, with and without the 5-min adaptive reference (`improve/review_r1_magnitude.py`). Both need the WESAD features. `make_vals_revision.py` collects every number they add to the paper into `vals_revision.json`, and `make_revision_figures.py` draws the new and redrawn figures. `run_all.sh` and `verify.py` include them. They were run in the original analysis environment on 31 October 2026 (logs in `logs/run_logs/review_R1_*.txt`); the end-to-end rerun of 14 October 2026 predates them.

## Citation

If you use this code, please cite the article above and this archive (https://doi.org/10.5281/zenodo.23438148). Citation metadata is in `CITATION.cff`.

## License

Code: MIT License (see `LICENSE`). The datasets remain under their original licences.

## Contact

Hamza Abu Owida (h.abuowida@ammanu.edu.jo) and Abdulmohsen Algarni (a.algarni@kku.edu.sa).
