# Hydrograph memory decides where a foundation model improves streamflow forecasts

Code, per-basin diagnostics and figures for **"Hydrograph memory decides where a foundation model
improves streamflow forecasts, and rainfall sets its ceiling"** (Yue Zhu, Qingyang Liu) — submitted
to *Environmental Research Letters*, 2026.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## What this is

A benchmark of two general-purpose time-series foundation models (**TimesFM**, **Chronos**) and a
non-hydrological Transformer control (**PatchTST**) for one-day-ahead daily streamflow forecasting
across **6,569 catchments** from six CAMELS-family archives on five continents (CAMELS-US, -BR, -CL,
-AUS, -IND, LamaH-CE), under one protocol: the most recent five years of each basin's record, every
model scored on identical one-day-ahead target points at a seven-day stride, against **lag-1
persistence**, a **fitted AR(2)** on the same 512-day context, and a **fairly trained global LSTM**.
All models are univariate (past discharge only), so the setting is *data-scarce*, not ungauged.

The paper's question is hydrological, not which model wins: **which environmental conditions decide
whether a pre-trained model adds skill?**

## Findings

1. **Hydrograph memory predicts zero-shot skill; catchment descriptors do not.** Spearman ρ between
   zero-shot NSE and lag-1 flow autocorrelation is +0.85 (TimesFM) and +0.86 (Chronos), n = 6,569;
   aridity and flow regulation are not significant. Static attributes supplied as covariates change
   skill by ΔNSE = +9×10⁻⁹.
2. **Where the models add value differs.** TimesFM beats persistence in 67.9 % of the flashiest
   memory quartile but only 21.2 % of the most damped, where it degrades a forecast that already
   attains NSE 0.97. Chronos improves on persistence modestly in every quartile
   (66.5 / 51.2 / 54.8 / 74.9 %), about as much as a fitted AR(2) does.
3. **The two large models are near-equal.** TimesFM has the higher median in five archives and
   Chronos in CAMELS-BR; the paired difference is significant only in CAMELS-US (TimesFM) and
   CAMELS-BR (Chronos). PatchTST has no skill anywhere.
4. **Rainfall sets the ceiling.** On the 1,230 basins with daily precipitation, TimesFM's residual is
   6.5× larger on rain days than dry days (Chronos 7.5×), in 97 % of basins, and the concentration
   strengthens as memory weakens. An oracle given the observed precipitation recovers only 8–28 % of
   the residual; LoRA fine-tuning and locally fitted forcings recover none of it.
5. A fairly trained global LSTM is the most accurate model where a multi-year local record exists;
   zero-shot TimesFM lands within 0.01–0.06 NSE of it with no target-basin training.

## Zero-shot results — median NSE (last five years, stride 7, identical targets)

| Dataset | Basins | Chronos | TimesFM | PatchTST | Persistence | AR(2) | LSTM (global, fair) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CAMELS-US  | 671   | 0.661 | **0.707** | −0.021 | 0.585 | 0.646 | 0.729 |
| CAMELS-BR  | 3,836 | **0.822** | 0.812 | −0.010 | 0.821 | 0.825 | 0.839 |
| CAMELS-CL  | 394   | 0.856 | **0.866** | −0.005 | 0.869 | 0.871 | 0.878 |
| CAMELS-AUS | 560   | 0.591 | **0.607** |  0.001 | 0.520 | 0.503 | 0.651 |
| LamaH-CE   | 880   | 0.674 | **0.677** |  0.000 | 0.705 | 0.703 | 0.732 |
| CAMELS-IND | 228   | 0.595 | **0.647** | −0.003 | 0.636 | 0.615 | 0.660 |

Bold = best foundation model per archive. Chronos point forecasts are the mean of **twenty sampled
trajectories** (`--num_samples 20`); an earlier single-draw configuration depressed Chronos's NSE
(e.g. 0.490 on CAMELS-US) and has been superseded — see `results/MANIFEST_LIVE_INPUTS.md`.

## Repository structure

```
models/
  run_foundation_models.py   zero-shot TimesFM / Chronos / PatchTST / persistence (--num_samples, --batch_size)
  run_ar_baseline.py         fitted AR(p) on each 512-day context, identical target points
  run_global_lstm.py         fair per-dataset global LSTM (per-basin z-score, matched targets)
  run_few_shot_lora.py       Chronos LoRA few-shot (--lora_rank)
  run_few_shot_multivariate.py, run_attribute_ablation.py, train_baselines.py, baselines.py
analysis/
  build_basin_attributes.py  flow signatures (scale-invariant) + archive attributes -> results/basin_attributes.csv
  hydrological_controls.py   Spearman / importance / memory-quartile analysis
  controls_diagnostics.py    circularity, ceiling (PSS), regression-to-mean, partial correlation
  archive_level_controls.py  archive-level replication and the CAMELS-CL analysis
  ar_baseline_analysis.py    TimesFM / Chronos vs fitted AR(2) by memory quartile
  rainfall_ceiling_test_v2.py, rainfall_ceiling_chronos.py, figure_rainfall_ceiling.py
  normalise_tail_rmse.py, physical_diagnostics.py, br_quality_subset_check.py, sensitivity_analysis.py
  figure_hydrological_controls.py, figure_zero_shot_overview.py, regenerate_l5s7_figures.py,
  figure_benchmark_design.py, build_real_basin_map.py
evaluation/
  metrics.py, assemble_table4_l5s7.py, eval_baselines_on_targets.py, reliability_diagrams.py, seasonal_bias.py
data/
  preprocess_camels.py, preprocess_camels_ind.py, fix_camels_br.py, dataloader.py
results/    per-basin diagnostics for every model x archive (`*_l5s7.csv`), the analysis reports
            (`*_REPORT.txt`, `CONTROLS_DIAGNOSTICS.txt`, ...) and MANIFEST_LIVE_INPUTS.md,
            which says which file backs which manuscript number
figures/    the manuscript and SI figures
```

## Setup

```bash
git clone https://github.com/zechariahyue/hydro-foundation-model.git
cd hydro-foundation-model
pip install -r requirements.txt
```

Paths come from environment variables with relative defaults:

```bash
export CAMELS_DATA_DIR="data/processed"   # per-basin parquet files (column QObs(mm/d) = native units, see paper)
export CAMELS_RAW_DIR="data/raw"          # raw archives (attribute tables, gauge coordinates)
export RESULTS_DIR="results"
export FIGURES_DIR="figures"
export CKPT_DIR="models/checkpoints"
```

Datasets (all public): CAMELS-US https://ral.ucar.edu/solutions/products/camels · CAMELS-BR
https://zenodo.org/records/3709337 · CAMELS-CL https://doi.org/10.5194/hess-22-5817-2018 ·
CAMELS-AUS https://doi.org/10.5194/essd-13-3847-2021 · CAMELS-IND https://doi.org/10.5194/essd-2024-560 ·
LamaH-CE https://zenodo.org/records/5153305.

## Reproducing the paper

```bash
DS="CAMELS-US CAMELS-BR CAMELS-CL CAMELS-AUS LamaH-CE CAMELS-IND"

# 1. preprocess
python data/preprocess_camels.py && python data/preprocess_camels_ind.py

# 2. zero-shot benchmark (TimesFM, Chronos with 20 samples, PatchTST, persistence)
python models/run_foundation_models.py --model all --mode zero_shot --datasets $DS \
       --last_years 5 --stride 7 --num_samples 20 --batch_size 8 --out_suffix _l5s7
python models/run_ar_baseline.py --datasets $DS --orders 1 2 5
python models/run_global_lstm.py --datasets $DS

# 3. hydrological controls (the paper's centrepiece)
python analysis/build_basin_attributes.py
python analysis/hydrological_controls.py
python analysis/controls_diagnostics.py
python analysis/archive_level_controls.py
python analysis/ar_baseline_analysis.py
python analysis/figure_hydrological_controls.py

# 4. rainfall ceiling (needs the CAMELS-US / CAMELS-AUS precipitation columns)
python analysis/rainfall_ceiling_test_v2.py
python analysis/rainfall_ceiling_chronos.py --stride 3 --num_samples 20
python analysis/figure_rainfall_ceiling.py

# 5. few-shot, ablation, diagnostics, tables and figures
python models/run_few_shot_lora.py --model chronos --fraction 0.10 --dataset CAMELS-IND --last_years 5
python models/run_attribute_ablation.py
python analysis/physical_diagnostics.py && python analysis/normalise_tail_rmse.py
python evaluation/assemble_table4_l5s7.py
python analysis/regenerate_l5s7_figures.py && python analysis/figure_zero_shot_overview.py
```

Every number in the manuscript traces to a file listed in `results/MANIFEST_LIVE_INPUTS.md`.

## Configuration

| Parameter | Value |
|---|---|
| Evaluation window / stride | most recent 5 years per basin / 7 days (all models) |
| Context / horizon | 512 days / 1 day |
| TimesFM | `google/timesfm-1.0-200m-pytorch` |
| Chronos | `amazon/chronos-t5-large`, 20 sampled trajectories per forecast (50 for reliability diagrams) |
| PatchTST | `ibm/patchtst-etth1-forecast` |
| AR(p) | intercept + p lags, OLS on each 512-day context, p ∈ {1, 2, 5} |
| LoRA | rank 8, α 16, q/v; fractions 0.01–0.25 (both few-shot arms scored on a single Chronos draw) |

## Citation

```bibtex
@article{zhu2026hydrograph_memory,
  title   = {Hydrograph memory decides where a foundation model improves streamflow forecasts,
             and rainfall sets its ceiling},
  author  = {Zhu, Yue and Liu, Qingyang},
  journal = {Environmental Research Letters},
  year    = {2026},
  note    = {submitted}
}
```

## License

MIT — see [LICENSE](LICENSE).
