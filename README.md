# exercise-stress-misclassification

Code, segment index and pre-specified analysis rules for:

> Abu Owida, H.; Alsarhan, A.; Banat, M.; Al-Fraihat, D.; Algarni, A. **Revealing Exercise-Induced Misclassification in Wearable Stress Detection for Automated Insulin Delivery.** *Sensors* (submitted, 2026).

The study re-evaluates stress and exercise recognition from Empatica E4 wrist signals under a pre-registered plan, measures how often a stress detector fires during structured exercise, and tests an exercise-robust detector on a sealed test set and on an external dataset (WESAD).

## Pre-registration

The analysis plan was registered on the Open Science Framework before any model was trained: https://osf.io/swge8

Rules for the analyses added after registration (episode-level evaluation, reduction of rest false alarms, external test) were fixed in dated documents whose SHA-256 hashes are reported in the paper (Table A3). Those documents are in `prespecifications/`; their hashes can be checked with `sha256sum`.

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
| `results/` | Archived outputs of the original run, and `vals.json`, every number quoted in the paper |
| `environment/` | Complete `pip freeze`, Python version and platform of the environment that produced the results |

### Pre-specification hashes (Table A3 of the paper)

| Document | SHA-256 |
|---|---|
| `episode_prespec.md` (episode-level evaluation) | `409b985cbfbb…` |
| `m6_prespec.md` (reduction of rest false alarms) | `2ef370e3330a…` |
| `wesad_addendum.md` (external test on WESAD) | `7d87c337fccf…` |

Each `.sha256` file is the original record: the hash on the first line and the time it was taken (UTC) on the second. Check them with `sha256sum prespecifications/*.md`. The scripts that use these rules refuse to run if a document has changed.

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

## Citation

If you use this code, please cite the article above and this archive (DOI to be added on release). Citation metadata is in `CITATION.cff`.

## License

Code: MIT License (see `LICENSE`). The datasets remain under their original licences.

## Contact

Hamza Abu Owida (h.abuowida@ammanu.edu.jo) and Abdulmohsen Algarni (a.algarni@kku.edu.sa).
