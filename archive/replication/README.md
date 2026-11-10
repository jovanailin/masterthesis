# Archived: replication of Tsantilis et al. (2025)

Replication of the 3 × 2 pooling study of Tsantilis, Giannopoulos, Dasaklis & Patsakis (2025), *Evaluating
cross-series pooling protocols in intermittent demand forecasting: A factorial study* (IEEE). It was
abandoned on 2026-11-04. The thesis now extends the paper with its own design instead of replicating it.
The paper remains the main related work.

The code is kept as it was when work stopped. It is not maintained, and it expects the original layout
(`notebooks/`, `src/`, `results/replication/`), so it does not run from this folder. Besides
`requirements.txt` it needs `lightgbm==4.6.0` and `torch==2.9.1`.

| Path | Contents |
|---|---|
| `02_replication.ipynb` | Sample selection and the analysis of the six cells |
| `src/experiment.py` | Tuning, top-3 ensembles and RMSSE of the six cells; resumable runner |
| `src/forecasters.py` | LightGBM, LSTM and echo state network direct 28-day forecasters |
| `results/` | Sample of 100 intermittent series, its allocation and selection funnel; partial results of the per-series LightGBM cell (47 of 100 series) |
| `figures/` | Sample over the regime cells |
| `predlozi_doprinos.txt` | First contribution proposals, written while the replication was still planned; superseded by the proposal in the repository root |

The modules still in use were kept in `src/`: `series_sampling.py` (stratified sample with Hamilton allocation)
and `factorial_stats.py` (Wilcoxon–Holm, Friedman–Nemenyi, critical difference diagram).
