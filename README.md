# An Open Benchmark of Time-Series Foundation Models for Streamflow Forecasting

Official code for **"An Open Benchmark of Time-Series Foundation Models for Zero-Shot and Few-Shot Streamflow Forecasting Across 6,569 Global Basins"** — submitted to *Journal of Hydrology*.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## Overview

We benchmark three time-series foundation models — **TimesFM**, **Chronos**, and **PatchTST** — for one-day-ahead daily streamflow forecasting across **6,569 catchments** from six CAMELS-family archives spanning five continents (CAMELS-US, -BR, -CL, -AUS, -IND, and LamaH-CE). The foundation models are used in their native **univariate** mode (past discharge only; no climate forcings or static attributes). Every model is evaluated under a **single consistent protocol** — the most recent five years of each basin's record, scored on identical one-day-ahead targets at a seven-day stride — against two baselines: **lag-1 persistence** and a **fairly trained, per-basin-standardised global LSTM** (the supervised model a data-rich agency would actually deploy).

**Key findings**

1. **TimesFM** is the strongest zero-shot model in all six datasets (median NSE 0.61–0.87), but its advantage over persistence is **regime-dependent**: significant in variable-flow basins (CAMELS-US, CAMELS-AUS), a tie in the perennial South-American basins (CAMELS-BR, CAMELS-CL), and below persistence in Central Europe (LamaH-CE).
2. **Chronos** does not significantly beat persistence in any region.
3. **PatchTST** shows negligible skill (median NSE ≈ 0), indicating that **pre-training domain coverage — not architecture alone — governs transfer**.
4. A **fairly trained global LSTM is the most accurate model overall**, but only where multi-year local records exist to train it. Zero-shot **TimesFM lands within 0.01–0.06 NSE** of that trained LSTM using **no** target-basin training data; the deployable value of foundation models is therefore **data efficiency** in basins where a supervised model cannot yet be built.
5. **Few-shot LoRA fine-tuning of Chronos yields no meaningful gain** over zero-shot (best ΔNSE within ±0.015), and is robust to LoRA rank.

---

## Results

### Zero-shot performance — median NSE (last-5-year window, stride-7, identical targets)

| Dataset | Basins | Chronos | TimesFM | PatchTST | Persistence | **LSTM (global, fair)** |
|---------|-------:|--------:|--------:|---------:|------------:|------------------------:|
| CAMELS-US  | 671   | 0.490 | 0.707 | −0.021 | 0.585 | **0.729** |
| CAMELS-BR  | 3,836 | 0.781 | 0.812 | −0.010 | 0.821 | **0.839** |
| CAMELS-CL  | 394   | 0.825 | 0.866 | −0.005 | 0.869 | **0.878** |
| CAMELS-AUS | 560   | 0.433 | 0.607 |  0.001 | 0.520 | **0.651** |
| LamaH-CE   | 880   | 0.629 | 0.677 |  0.000 | 0.705 | **0.732** |
| CAMELS-IND | 228   | 0.418 | 0.647 | −0.003 | 0.636 | **0.660** |

Total **6,569** basins. Bold = best model per dataset (the fair global LSTM is strongest in every dataset, but requires a multi-year local training record). A *naive-transfer* LSTM (CAMELS-US weights + source normalisation) collapses across regions (median NSE −0.2 to −302,970) — a pure normalisation artefact, not a property of supervised models.

### Few-shot fine-tuning — Chronos LoRA

Under the matched protocol, few-shot LoRA provides **no meaningful improvement** over zero-shot on any of the three datasets tested (CAMELS-BR, -AUS, -IND): the best fraction stays within **±0.015 NSE** of the zero-shot baseline, and the response is flat across data fractions. A from-scratch multivariate LSTM (discharge + forcings) trained on the same local data does *worse* still. The foundation model's pre-trained prior — not local adaptation — is what delivers skill in data-scarce basins.

### Sensitivity analyses (all robust)

| Analysis | Result |
|----------|--------|
| Context-window length (TimesFM, CAMELS-US) | NSE 0.604 / 0.562 / 0.692 / 0.707 at 90 / 180 / 365 / 512 days — even 90 days beats persistence (0.585) |
| LoRA rank (Chronos, CAMELS-IND, f=0.10) | NSE within −0.03…+0.01 of zero-shot at ranks 4 / 8 / 16 / 32 — no gain at any rank |
| CAMELS-BR station quality | Restricting to the 897 quality-selected catchments shifts every model ≤0.007 NSE; ordering unchanged |

---

## Repository Structure

```
foundation_model/
├── models/
│   ├── baselines.py                  # LSTM / ConvLSTM architectures
│   ├── train_baselines.py            # Train LSTM/ConvLSTM (naive-transfer baseline)
│   ├── run_foundation_models.py      # Zero-shot TimesFM, Chronos, PatchTST inference
│   ├── run_global_lstm.py            # Fair per-dataset global LSTM (per-basin z-score, matched targets)
│   ├── run_few_shot_lora.py          # Chronos LoRA few-shot fine-tuning (--lora_rank knob)
│   ├── run_few_shot_multivariate.py  # From-scratch uni/multivariate few-shot LSTM
│   └── run_attribute_ablation.py     # TimesFM in-context covariate ablation (static attrs / forcings)
│
├── evaluation/
│   ├── metrics.py                    # NSE, KGE, RMSE, PBIAS, FDC-KGE, tail-RMSE, CRPS
│   ├── compute_bootstrap_stats.py    # Bootstrap CIs + Wilcoxon (Benjamini–Hochberg)
│   ├── eval_baselines_on_targets.py  # Evaluate trained LSTM on target basins
│   ├── reliability_diagrams.py       # Chronos probabilistic reliability by flow regime
│   ├── seasonal_bias.py              # Seasonal performance decomposition
│   └── generate_figures.py           # Reproduce paper figures
│
├── analysis/
│   ├── physical_diagnostics.py       # FDC-KGE, PBIAS physical metrics
│   ├── sensitivity_analysis.py       # Context-length sensitivity
│   ├── build_real_basin_map.py       # Per-basin geographic / spatial-skill maps
│   ├── compute_statistics.py         # Summary statistics
│   └── generate_figures.py           # Analysis figures
│
├── data/
│   ├── dataloader.py                 # Multi-basin dataset loader
│   ├── preprocess_camels.py          # Preprocess CAMELS-US/BR/CL/AUS/LamaH-CE
│   └── preprocess_camels_ind.py      # Preprocess CAMELS-IND
│
├── configs/experiment_config.yaml    # Hyperparameters, protocol, model checkpoints
├── results/    # generated CSVs    (gitignored)
├── figures/    # generated figures (gitignored)
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/zechariahyue/hydro-foundation-model.git
cd hydro-foundation-model
pip install -r requirements.txt
```

Paths are read from environment variables (with sensible defaults), so nothing is hard-coded:

```bash
export CAMELS_DATA_DIR="/path/to/processed/camels"  # per-basin parquet files (default: data/processed)
export CAMELS_RAW_DIR="/path/to/raw/camels"         # raw archives, for the basin-map script (default: data/raw)
export RESULTS_DIR="results"                         # output CSVs
export FIGURES_DIR="figures"                         # output figures
```

### Datasets (public)

| Dataset | Source |
|---------|--------|
| CAMELS-US  | https://ral.ucar.edu/solutions/products/camels |
| CAMELS-BR  | https://zenodo.org/records/3709337 |
| CAMELS-CL  | https://doi.org/10.5194/hess-22-5817-2018 |
| CAMELS-AUS | https://doi.org/10.5194/essd-13-3847-2021 |
| CAMELS-IND | https://doi.org/10.5194/essd-2024-560 |
| LamaH-CE   | https://zenodo.org/records/5153305 |

Basin counts in the results table are the number actually evaluated on the five-year window (basins with a sufficient record).

---

## Reproducing Paper Results

```bash
# 1. Preprocess
python data/preprocess_camels.py
python data/preprocess_camels_ind.py

# 2. Zero-shot foundation models (TimesFM, Chronos, PatchTST) + persistence
python models/run_foundation_models.py --model all --mode zero_shot \
       --datasets CAMELS-US CAMELS-BR CAMELS-CL CAMELS-AUS CAMELS-IND LamaH-CE \
       --last_years 5 --stride 7 --out_suffix _l5s7

# 3. Fair global-LSTM baseline (matched basins/targets)
python models/run_global_lstm.py --datasets CAMELS-US CAMELS-BR CAMELS-CL CAMELS-AUS CAMELS-IND LamaH-CE

# 4. Few-shot Chronos LoRA  (sweep --lora_rank for the rank sensitivity)
for FRAC in 0.01 0.05 0.10 0.25; do
  for DS in CAMELS-BR CAMELS-AUS CAMELS-IND; do
    python models/run_few_shot_lora.py --model chronos --fraction $FRAC --dataset $DS --last_years 5
  done
done

# 5. Ablations, diagnostics, sensitivity, maps
python models/run_attribute_ablation.py
python models/run_few_shot_multivariate.py
python evaluation/reliability_diagrams.py
python analysis/physical_diagnostics.py
python analysis/sensitivity_analysis.py
python analysis/build_real_basin_map.py

# 6. Statistics and figures
python evaluation/compute_bootstrap_stats.py
python evaluation/generate_figures.py
```

---

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Evaluation window | most recent 5 years of each basin's record (per-basin) |
| Evaluation stride | 7 days (uniform across all models) |
| Context length | 512 days |
| Forecast horizon | 1 day |
| TimesFM | `google/timesfm-1.0-200m-pytorch` |
| Chronos | `amazon/chronos-t5-large` (20 samples for point forecasts; 50 for reliability) |
| PatchTST | `ibm/patchtst-etth1-forecast` |
| LoRA | rank 8, α 16, target modules q/v |
| Few-shot fractions | 0.01, 0.05, 0.10, 0.25 |

---

## Citation

```bibtex
@article{zhu2025streamflow_benchmark,
  title   = {An Open Benchmark of Time-Series Foundation Models for Zero-Shot
             and Few-Shot Streamflow Forecasting Across 6,569 Global Basins},
  author  = {Zhu, Yue and Liu, Qingyang},
  journal = {Journal of Hydrology},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
