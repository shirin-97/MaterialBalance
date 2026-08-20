# Havlena-Odeh Reservoir Drive Mechanism Diagnostic Tool
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://materialbalance-3nmsrpvcjnthpowbh7v7z6.streamlit.app/)

A Streamlit app that implements the Havlena-Odeh straight-line method to
diagnose a reservoir's primary drive mechanism — depletion, gas cap, water
drive, or a combination — from real, messy field production and PVT data.

## What This Does

Upload a production file and a PVT file (any reasonable column names, `.csv`
or `.xlsx`), map your columns to the physical parameters the engine needs,
and the app runs the full Havlena-Odeh material balance: underground
withdrawal (F), the expansion terms (Eo, Eg, Efw), the classic sequential
straight-line diagnostics (F vs Eo, F vs Et, water-drive residual), and a
jointly-fit (NNLS) estimate of OOIP, gas-cap ratio, and aquifer constant with
non-negative drive indices (DDI/SDI/WDI) that sum to ~100%.

## Key Features

- **Guided 3-step data intake** (Tab 1): upload → optional well filtering
  (auto-detected for multi-well files) → column mapping, with each step
  gated behind the last.
- **Dynamic column mapping**: your file's headers rarely match the engine's
  names exactly — map them once via dropdowns, smart-defaulted from common
  naming conventions.
- **Metric ↔ Imperial conversion**: production data in Sm3/Bar (e.g. Volve/
  NPD-style exports) is converted to STB/Scf/psia automatically.
- **Incremental-vs-cumulative detection**: auto-detects whether your Np
  column is a daily/monthly rate or a running total, and converts if needed.
- **Daily pressure noise filtering**: a 30-day rolling average smooths
  choke/gauge noise out of daily downhole pressure before it reaches the
  physics engine.
- **8-point data validation** with clear error/warning banners before
  analysis ever runs.
- **4 analysis tabs**: production diagnostics, the 3 classic H-O
  straight-line plots with a rewired drive-mechanism summary card, and a
  drive-index breakdown (DDI/SDI/WDI) with a 100%-stacked-area chart.

## How to Use

1. **Tab 1 — Data Input**: upload a production file and a PVT file. If your
   production file has multiple wells, pick one. Map your columns to the
   required physical parameters, set the units and incremental/cumulative
   toggle (both auto-suggested), and click **Confirm Mapping & Run
   Analysis**.
2. **Tab 2 — Production Diagnostics**: review production history, GOR/WOR
   trends, and pressure decline before trusting the H-O analysis.
3. **Tab 3 — Havlena-Odeh Analysis**: the three classic diagnostic plots
   (each with its governing equation shown), plus the Drive Mechanism
   Summary Card (OOIP, gas-cap ratio, aquifer constant, primary mechanism —
   all from the joint NNLS fit). The sidebar's `m` slider is for visually
   testing the F-vs-Et straight line only; it doesn't feed the summary card.
4. **Tab 4 — Insights & Drive Indices**: the same joint fit expressed as a
   depletion/gas-cap/water-drive percentage breakdown over time.

See `integration_matrix.md` for a worked example of how badly a
depletion-only fit misreads gas-cap and water-drive data, and how much the
gas-cap and water-drive diagnostics recover.

## Data Source

`data/sample_production_data/` includes a real multi-well production export
and a representative PVT table for trying the tool out:

- **Volve field production data** — released by Equinor (operator), with
  ExxonMobil, Bayerngas, and the Norwegian Petroleum Directorate (NPD), under
  **CC BY-NC-SA 4.0** (attribution required, non-commercial, share-alike;
  the data may not be resold). See `data/sample_production_data/license.txt`
  for the full terms, and https://www.equinor.com/energy/volve-data-sharing
  for the original release.
- `dummy_pvt.xlsx` / `dummy_pvt_metric.xlsx` — representative PVT tables
  (Imperial and Metric units) for pairing with the Volve production data,
  since Volve's public release doesn't include a lab PVT report.

## Project Structure

```
app.py                   Streamlit UI — data intake wizard + 4 analysis tabs
src/
  pvt.py                 Standing's correlations, phase boundary, PVT interpolation
  material_balance.py    F/Eo/Eg/Efw/Et, zero-intercept regression, NNLS joint fit
  synthetic.py            Synthetic field-case generator (used by tests only —
                          no longer exposed in the UI; see the spec changelog)
  validation.py           8-point validation + real-data sanitization pipeline
  plots.py                All Plotly figure builders
tests/                    pytest suite (67 tests as of this writing)
data/
  sample_production_data/ Real Volve production data + PVT templates
  sample_templates/       Minimal CSV templates for the upload flow
integration_matrix.md     Worked verification of the drive-mechanism diagnostics
```

## Physics Notes

- PVT: Standing's correlations (saturated region only); undersaturated
  region uses constant isothermal compressibility.
- Aquifer model: simplified steady-state (We = C·(Pi−P)), not
  Van Everdingen-Hurst.
- Classic H-O straight lines (Tab 3's three plots): zero-intercept
  least-squares via numpy, no scipy.
- Drive indices / summary card (Tab 3 card, Tab 4): joint N/m/C fit via
  `scipy.optimize.nnls` — the sequential single-variable fits compound error
  and can go negative on collinear data; NNLS constrains all three to ≥ 0.
- All H-O straight lines forced through the origin — physically required.
- Internal units throughout the physics engine: psia, STB, Scf, °F.

## Run Locally

```
pip install -r requirements.txt
python -m streamlit run app.py
```

If `streamlit run app.py` isn't found on your PATH, use
`python -m streamlit run app.py` instead — it works regardless of PATH.

## Run Tests

```
pip install pytest
python -m pytest tests/ -v
```

## Deploying

Push this repo to GitHub, then connect it at https://share.streamlit.io
with the main file path set to `app.py`.
