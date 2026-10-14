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
| `code/` | Analysis scripts: integrity checks, segment mapping, windowing and features, models, evaluation (E1–E6), exploratory and detector analyses, statistics, figures |
| `segment_index/` | Mapping of every event mark to a protocol block (885 blocks) |
| `prespecifications/` | Hashed rule documents for the episode-level, false-alarm and external analyses |
| `logs/` | Trial log of all configurations evaluated during detector development |

## Environment

Analyses were run in Python 3.13 with scikit-learn 1.9, XGBoost 3.4, statsmodels 0.15 and NeuroKit2 0.2.13 (full list in `requirements.txt`). Seeds: 20261008 (registered analyses) and 20261009 (detector work).

```bash
pip install -r requirements.txt
```

## Citation

If you use this code, please cite the article above and this archive (DOI to be added on release). Citation metadata is in `CITATION.cff`.

## License

Code: MIT License (see `LICENSE`). The datasets remain under their original licences.

## Contact

Hamza Abu Owida (h.abuowida@ammanu.edu.jo) and Abdulmohsen Algarni (a.algarni@kku.edu.sa).
