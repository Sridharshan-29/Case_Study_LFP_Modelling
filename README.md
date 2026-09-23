# A123 ANR26650M1B - DFN Cell Physical Modeling & Parametrization in PyBaMM

This repo builds a DFN model of the A123 ANR26650M1B (2.5 Ah, LFP/graphite) cell in PyBaMM,
fits it to measured charge/discharge/HPPC data, optimizes its kinetic and transport
parameters, and validates the result on the HPPC test (never used for fitting).

See `REPORT.pdf` for the full write-up (methodology, results, plots, discussion) for every step.

## Folder structure

```
01_STEP_1_AND_2_HPPC/                          Step 1 & 2: baseline DFN model, run on HPPC

02_STEP_3_OCP_AND_ANALYSIS/                    Step 3: literature OCP + fitted stoichiometry
                                               window + capacity matching, run on HPPC and
                                               on the charge/discharge sheets, residual analysis

03_STEP_4_AND_5_OPTIMIZATION_VALIDATION/       Step 4: parameter optimization
                                               Step 5: validation with the optimized parameters

04_REQUIRED_PLOTS/                             Final plots required by the assignment

requirements.txt                               Python dependencies

REPORT.pdf                                     Full report

Case Study Description.pdf                     Description of the Case Study
```

Each numbered folder is self-contained: script(s) + `inputs/` + `results/`. Every script finds
its own input files and writes its own outputs automatically, based on its own location on disk
(`Path(__file__).resolve().parent`) - so it does not matter which directory you run it from.

## Setup

```bash
pip install -r requirements.txt
```

## Data files - put these in every `inputs/` folder that needs them

- `All_MAT_Files_Converted.xlsx` (Charge_1C, Charge_2C, Charge_3C, Charge_4C, Discharge_1C, HPPC sheets)
- `OCV_25C_Clean_Monotonic.xlsx` (measured OCV vs SOC lookup table obtained from the external datasets)

| Folder | Needs |
|---|---|
| `01_STEP_1_AND_2_HPPC/inputs/` | `All_MAT_Files_Converted.xlsx` |
| `02_STEP_3_OCP_AND_ANALYSIS/inputs/` | both files |
| `03_STEP_4_AND_5_OPTIMIZATION_VALIDATION/inputs/` | both files |

(The same two files are just copied into each folder that needs them, so each step is fully
self-contained and can be run/inspected independently.)

## How to run - order matters

### Step 1 & 2 - baseline model - The Base Simulation (DFN model simulation) asked in the Assignment

```bash
cd 01_STEP_1_AND_2_HPPC
python DFN model simulation.py
```

Builds the plain Prada2013 DFN model (capacity-scaled to 2.5 Ah only, default OCP curves, no
tuning) and runs it on the HPPC current profile.
Outputs -> `results/`: `HPPC_DFN_Simulation_Results.xlsx`, `task2_error_metrics.json`, 2 plots.

### Step 3 - literature OCP + stoichiometry/capacity fit + residual analysis

```bash
cd 02_STEP_3_OCP_AND_ANALYSIS
python step3_safari_ocp.py             # HPPC, with the fixed model
python step3_charge_discharge.py       # same fixed model, run on all 5 charge/discharge sheets
```

Run `step3_safari_ocp.py` first - it prints the fitted stoichiometry window, which
`step3_charge_discharge.py` reproduces internally (both scripts fit it independently from the
same OCV table, so results match; there is no file handoff needed between them).
Outputs -> `results/`: `HPPC_DFN_Simulation_Results_SafariOCP.xlsx`,
`Simulation_Results_All_Sheets.xlsx`, `task3_error_metrics_safari_ocp.json`,
`all_sheets_error_metrics.json`, plus the OCV-fit, electrode-OCP, residual and per-sheet plots.

### Step 4 - optimization

```bash
cd 03_STEP_4_AND_5_OPTIMIZATION_VALIDATION
python DFN parameter optimization.py
```

Runs a sensitivity screen, a Latin-hypercube global sample, then a gradient-based local
refinement, on the 5 charge/discharge sheets. Saves progress as it goes (in
`results/step4_state/`), so if it is interrupted, running the same command again resumes
instead of starting over. Add `--reset` to discard saved progress and start clean.
Output -> `results/optimized_parameters.json` (the final parameter set - **required before
Step 5 can run**).

### Step 5 - validation

```bash
python DFN model validation.py
```

Must be run **after** Step 4, from the same folder (it reads
`results/optimized_parameters.json`). Applies the optimized parameters with
`parameter_values.update()`, re-runs all 5 training sheets plus the HPPC test (HPPC was never
used to fit anything - this is the real validation), and plots the internal physical variables
(anode/cathode potential vs Li/Li+, surface stoichiometry) the assignment asks for.
Outputs -> `results/`: `Validation_Results.xlsx`, `validation_metrics.json`, `fig_val_*.png`.

**Summary of the required order:**
```
DFN model simulation.py
        |
step3_safari_ocp.py  -->  step3_charge_discharge.py
        |
DFN parameter optimization.py  -->  optimized_parameters.json  -->  DFN model validation.py
```

## 04_REQUIRED_PLOTS

It containts the Final plots required by the assignment
1) Measured vs simulated voltage in step 2
2) Measured vs simulated voltage in step 5(w.r.t. training (HPPC) data)
3) Important physical variables_hppc
4) Important physical variables_Charge_4C

## Notes on the model

- Model: PyBaMM `lithium_ion.DFN`, isothermal, Fickian particle diffusion.
- Base parameter set: `Prada2013`, capacity-scaled from 2.3 Ah to 2.5 Ah via electrode width.
- From Step 3 onward: cathode/anode OCP replaced with literature Safari & Delacourt (2011)
  functions for a graphite/LFP cell, with a stoichiometry window fitted to the cell's own
  measured OCV table, and the electrode active-material fractions rebalanced so both
  electrodes deliver the target 2.5 Ah over that fitted window.
- Step 4 optimizes: negative/positive exchange-current density, negative/positive particle
  diffusivity, electrode/electrolyte conductivity, electrolyte diffusivity, contact
  resistance, and a small physically-constrained correction to the cathode OCP curve.
  Parameters the data cannot identify are frozen automatically (see
  `optimized_parameters.json -> frozen_not_identified`).
- Full methodology, justification for every choice, and discussion of the results are in
  `REPORT.pdf`.
