import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_XLSX = INPUT_DIR / "All_MAT_Files_Converted.xlsx"
OCV_XLSX = INPUT_DIR / "OCV_25C_Clean_Monotonic.xlsx"
OUT_DIR = OUTPUT_DIR

RUN_TAG = f"{int(time.time() * 1000)}"

FARADAY = 96485.33212  
TARGET_CAPACITY_AH = 2.5
NEG_DIFFUSIVITY = 2.5e-14  

# =============================================================================
# 1. SAFARI & DELACOURT (2011) OCP FUNCTIONS - TABLE I, SIGN-CORRECTED
# =============================================================================
def positive_electrode_ocp_safari(sto):
    """LFP cathode OCP, U_p(y) vs Li/Li+, Safari & Delacourt 2011 Table I."""
    y = sto
    return (
        3.4323
        - 0.8428 * pybamm.exp(-80.2493 * (1 - y) ** 1.3198)
        - 3.2474e-6 * pybamm.exp(20.2645 * (1 - y) ** 3.8003)
        + 3.2482e-6 * pybamm.exp(20.2646 * (1 - y) ** 3.7995)
    )

def negative_electrode_ocp_safari(sto):
    """Graphite anode OCP, U_n(x) vs Li/Li+, Safari & Delacourt 2011 Table I."""
    x = sto
    return (
        0.6379
        + 0.5416 * pybamm.exp(-305.5309 * x)
        + 0.044 * pybamm.tanh(-(x - 0.1958) / 0.1088)
        - 0.1978 * pybamm.tanh((x - 1.0571) / 0.0854)
        - 0.6875 * pybamm.tanh((x + 0.0117) / 0.0529)
        - 0.0175 * pybamm.tanh((x - 0.5692) / 0.0875)
    )

def _up_np(y):
    return (
        3.4323
        - 0.8428 * np.exp(-80.2493 * (1 - y) ** 1.3198)
        - 3.2474e-6 * np.exp(20.2645 * (1 - y) ** 3.8003)
        + 3.2482e-6 * np.exp(20.2646 * (1 - y) ** 3.7995)
    )

def _un_np(x):
    return (
        0.6379
        + 0.5416 * np.exp(-305.5309 * x)
        + 0.044 * np.tanh(-(x - 0.1958) / 0.1088)
        - 0.1978 * np.tanh((x - 1.0571) / 0.0854)
        - 0.6875 * np.tanh((x + 0.0117) / 0.0529)
        - 0.0175 * np.tanh((x - 0.5692) / 0.0875)
    )

# =============================================================================
# 2. DATA LOADING & CLEANING UTILITIES
# =============================================================================
def load_sheet_data(sheet_name: str, path: str = RAW_XLSX) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.rename(
        columns={"Time": "t", "Voltage": "V", "Current": "I_raw"}
    )
    df = df.sort_values("t").reset_index(drop=True)
    keep = np.concatenate(([True], np.diff(df["t"].values) > 0))
    df = df.loc[keep].reset_index(drop=True)
    df["t"] = df["t"] - df["t"].iloc[0]
    df["I_pybamm"] = -df["I_raw"]
    return df[["t", "V", "I_raw", "I_pybamm"]]

def decimate_rests(df: pd.DataFrame, rest_dt_target: float = 5.0, current_tol: float = 1e-3) -> pd.DataFrame:
    is_rest = df["I_raw"].abs().values < current_tol
    keep = np.ones(len(df), dtype=bool)
    i = 0
    n = len(df)
    while i < n:
        if not is_rest[i]:
            i += 1
            continue
        j = i
        while j < n and is_rest[j]:
            j += 1
        t_block = df["t"].values[i:j]
        if len(t_block) > 2:
            block_keep = np.zeros(len(t_block), dtype=bool)
            block_keep[0] = True
            block_keep[-1] = True
            last_t = t_block[0]
            for k in range(1, len(t_block) - 1):
                if t_block[k] - last_t >= rest_dt_target:
                    block_keep[k] = True
                    last_t = t_block[k]
            keep[i:j] = block_keep
        i = j
    return df.loc[keep].reset_index(drop=True)

# =============================================================================
# 3. MODEL DEFINITION & BASE PARAMETERIZATION
# =============================================================================
model = pybamm.lithium_ion.DFN(
    options={
        "thermal": "isothermal",
        "surface form": "false",
        "particle": "Fickian diffusion",
    }
)

parameter_values = pybamm.ParameterValues("Prada2013")

print("=" * 70)
print("Prada2013 default parameters relevant to cell size / capacity:")
for k in [
    "Nominal cell capacity [A.h]", "Current function [A]", "Electrode height [m]",
    "Electrode width [m]", "Lower voltage cut-off [V]", "Upper voltage cut-off [V]",
]:
    print(f"  {k}: {parameter_values[k]}")
print("=" * 70)

capacity_scale = 2.5 / 2.3
parameter_values.update(
    {
        "Nominal cell capacity [A.h]": 2.5,
        "Electrode width [m]": parameter_values["Electrode width [m]"] * capacity_scale,
        "Lower voltage cut-off [V]": 2.0,
        "Upper voltage cut-off [V]": 3.8,
    }
)

parameter_values["Negative electrode OCP [V]"] = negative_electrode_ocp_safari
parameter_values["Positive electrode OCP [V]"] = positive_electrode_ocp_safari

# =============================================================================
# 4. FIT THE STOICHIOMETRY WINDOW TO THE MEASURED FULL-CELL OCV
# =============================================================================
ocv_df = pd.read_excel(OCV_XLSX, sheet_name="Lookup_Table_25C")
soc_meas = ocv_df["SOC_%"].to_numpy() / 100.0
ocv_meas = ocv_df["OCV_V"].to_numpy()
sort_idx = np.argsort(soc_meas)
soc_meas, ocv_meas = soc_meas[sort_idx], ocv_meas[sort_idx]

def model_ocv_np(params, soc):
    x0, x100, y0, y100 = params
    x = x0 + soc * (x100 - x0)
    y = y0 + soc * (y100 - y0)
    return _up_np(y) - _un_np(x)

def residuals(params, soc, v_meas):
    return model_ocv_np(params, soc) - v_meas

p0 = [0.00, 0.80, 0.86, 0.03]
bounds_lower = [0.00, 0.65, 0.55, 0.00]
bounds_upper = [0.20, 0.95, 0.95, 0.30]

result = least_squares(
    residuals, p0, args=(soc_meas, ocv_meas), bounds=(bounds_lower, bounds_upper)
)
x_0, x_100, y_0, y_100 = result.x
fit_rmse = float(np.sqrt(np.mean(residuals(result.x, soc_meas, ocv_meas) ** 2)))

print("=" * 70)
print("Stoichiometry window fitted to measured OCV (Safari & Delacourt OCPs):")
print(f"  x_0={x_0:.4f}  x_100={x_100:.4f}  y_0={y_0:.4f}  y_100={y_100:.4f}")
print(f"  Fit RMSE across full 0-100% SOC range: {fit_rmse * 1000:.2f} mV")
print("=" * 70)

check_soc = np.linspace(0, 1, 1001)
check_ocv = model_ocv_np(result.x, check_soc)
n_violations = int(np.sum(np.diff(check_ocv) < 0))
print(f"PRE-FLIGHT CHECK: model OCV monotonicity violations: {n_violations} / 1000")
if n_violations > 0:
    raise RuntimeError(f"Fitted model OCV non-monotonic at {n_violations} points - abort.")
print("  Monotonicity: OK - proceeding.")
print("=" * 70)

# =============================================================================
# 4b. CAPACITY MATCHING
# =============================================================================
def window_capacity_ah(pv, electrode: str, sto_span: float) -> float:
    area = pv["Electrode height [m]"] * pv["Electrode width [m]"]
    eps = pv[f"{electrode} electrode active material volume fraction"]
    thick = pv[f"{electrode} electrode thickness [m]"]
    c_max = pv[f"Maximum concentration in {electrode.lower()} electrode [mol.m-3]"]
    return eps * thick * area * c_max * FARADAY / 3600.0 * sto_span


span_n = x_100 - x_0
span_p = y_0 - y_100
Q_n_before = window_capacity_ah(parameter_values, "Negative", span_n)
Q_p_before = window_capacity_ah(parameter_values, "Positive", span_p)

eps_n_old = parameter_values["Negative electrode active material volume fraction"]
eps_p_old = parameter_values["Positive electrode active material volume fraction"]
eps_n_new = eps_n_old * TARGET_CAPACITY_AH / Q_n_before
eps_p_new = eps_p_old * TARGET_CAPACITY_AH / Q_p_before

for name, eps_s, eps_l in [
    ("Negative", eps_n_new, parameter_values["Negative electrode porosity"]),
    ("Positive", eps_p_new, parameter_values["Positive electrode porosity"]),
]:
    if eps_s + eps_l > 1.0:
        print(f"  WARNING: {name} eps_s + porosity = {eps_s + eps_l:.3f} > 1 (unphysical)")

parameter_values["Negative particle diffusivity [m2.s-1]"] = NEG_DIFFUSIVITY
print(f"Negative particle diffusivity set to: {NEG_DIFFUSIVITY:.2e} m2/s")
parameter_values.update(
    {
        "Negative electrode active material volume fraction": eps_n_new,
        "Positive electrode active material volume fraction": eps_p_new,
    }
)
Q_n_after = window_capacity_ah(parameter_values, "Negative", span_n)
Q_p_after = window_capacity_ah(parameter_values, "Positive", span_p)

print("=" * 70)
print(f"CAPACITY MATCHING (target {TARGET_CAPACITY_AH:.3f} Ah over fitted window)")
print(f"  Negative: {Q_n_before:.3f} Ah -> {Q_n_after:.3f} Ah "
      f"(eps_n {eps_n_old:.4f} -> {eps_n_new:.4f}, x{eps_n_new / eps_n_old:.4f})")
print(f"  Positive: {Q_p_before:.3f} Ah -> {Q_p_after:.3f} Ah "
      f"(eps_p {eps_p_old:.4f} -> {eps_p_new:.4f}, x{eps_p_new / eps_p_old:.4f})")
print("=" * 70)

# =============================================================================
# 5. SOLVE DFN MODEL FOR ALL 5 SHEETS
# =============================================================================
sheets = ["Discharge_1C", "Charge_1C", "Charge_2C", "Charge_3C", "Charge_4C"]
excel_output_path = os.path.join(OUT_DIR, "Simulation_Results_All_Sheets.xlsx")
excel_writer = pd.ExcelWriter(excel_output_path, engine="openpyxl")

soc_from_model_ocv = interp1d(check_ocv, check_soc, kind="linear", fill_value="extrapolate")
all_metrics = {}

try:
    solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
    solver_name = "IDAKLUSolver"
except Exception as exc:
    print(f"IDAKLU unavailable ({exc}); falling back to CasadiSolver(dt_max=10)")
    solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-6, dt_max=10)
    solver_name = "CasadiSolver(dt_max=10)"

print(f"Solver: {solver_name}")

for sheet_name in sheets:
    print("\n" + "=" * 70)
    print(f"PROCESSING SHEET: {sheet_name}")
    print("=" * 70)

    # 5.1 Load sheet data
    sheet_df = load_sheet_data(sheet_name)
    v_initial = float(sheet_df["V"].iloc[0])
    print(f"First voltage measured in {sheet_name}: {v_initial:.4f} V")

    # 5.2 Backtrack Initial SOC
    initial_soc = float(np.clip(soc_from_model_ocv(v_initial), 0.0, 1.0))
    print(f"Backtracked Initial SoC: {initial_soc:.4f} ({initial_soc * 100:.2f}%)")

    x_init = x_0 + initial_soc * (x_100 - x_0)
    y_init = y_0 + initial_soc * (y_100 - y_0)

    param_run = parameter_values.copy()
    param_run.update(
        {
            "Initial concentration in negative electrode [mol.m-3]": (
                x_init * param_run["Maximum concentration in negative electrode [mol.m-3]"]
            ),
            "Initial concentration in positive electrode [mol.m-3]": (
                y_init * param_run["Maximum concentration in positive electrode [mol.m-3]"]
            ),
        }
    )

    # 5.3 Prepare Solver Input
    sheet_solve = decimate_rests(sheet_df, rest_dt_target=5.0)
    t_data = sheet_solve["t"].to_numpy()
    i_data = sheet_solve["I_pybamm"].to_numpy()

    current_interpolant = pybamm.Interpolant(
        t_data, i_data, pybamm.t, name=f"Current_interpolant_{sheet_name}_{RUN_TAG}"
    )
    param_run["Current function [A]"] = current_interpolant

    # 5.4 Solve DFN Model
    sim = pybamm.Simulation(model, parameter_values=param_run, solver=solver)

    print(f"Solving DFN model for {sheet_name} ({t_data[-1]:.0f} s / {t_data[-1]/3600:.2f} h)...")
    t0 = time.time()
    try:
        solution = sim.solve(t_eval=t_data)
    except pybamm.SolverError:
        solution = sim.solution

    print(f"Solve completed in {time.time() - t0:.1f} s | Termination: {solution.termination}")

    # 5.5 Post-Processing & Excel Writing
    t_full = sheet_df["t"].to_numpy()
    v_meas_full = sheet_df["V"].to_numpy()
    sim_t = solution["Time [s]"].entries
    sim_v = solution["Terminal voltage [V]"].entries

    mask = t_full <= sim_t[-1]
    v_sim_interp = np.interp(t_full[mask], sim_t, sim_v)
    v_meas = v_meas_full[mask]
    t_eval_full = t_full[mask]
    i_meas = sheet_df["I_raw"].to_numpy()[mask]
    residual = v_sim_interp - v_meas

    # ------------------------------------------------------------------
    # 5.5a EARLY-TERMINATION DIAGNOSTIC 
    # ------------------------------------------------------------------
    i_full = sheet_df["I_raw"].to_numpy()          
    loaded = np.abs(i_full) > 1e-3
    t_load_end = float(t_full[loaded][-1]) if loaded.any() else float(t_full[-1])
    t_sim_end = float(sim_t[-1])
    coverage_loaded = float(min(t_sim_end, t_load_end) / t_load_end)

    q_required_ah = float(abs(np.trapezoid(i_full, t_full)) / 3600.0) if hasattr(np, "trapezoid") \
        else float(abs(np.trapz(i_full, t_full)) / 3600.0)          # net Ah the test moves
    is_discharge = sheet_name.lower().startswith("discharge")
    q_model_avail_ah = (initial_soc if is_discharge else (1.0 - initial_soc)) * TARGET_CAPACITY_AH

    print(f"  Model end: {t_sim_end:.0f} s | last loaded sample in data: {t_load_end:.0f} s "
          f"| data end: {t_full[-1]:.0f} s")
    print(f"  Charge the test moves (measured current): {q_required_ah:.3f} Ah | "
          f"model can deliver from this start SOC: {q_model_avail_ah:.3f} Ah")

    if t_sim_end < t_full[-1] - 1.0:
        tail = t_full > t_sim_end
        tail_load = loaded & tail
        print(f"  WARNING: simulation stopped early ({solution.termination}).")
        print(f"    Measured samples after sim end: {int(tail.sum())} "
              f"({int(tail_load.sum())} of them under load)")
        print(f"    Measured V in that tail: {v_meas_full[tail].max():.3f} -> {v_meas_full[tail][-1]:.3f} V, "
              f"|I| max {np.abs(i_full[tail]).max():.3f} A")
        if tail_load.any():
            print("    -> The cell is still being loaded after the model is empty/full: "
                  "model capacity or end-of-range kinetics is short of the real cell.")
        else:
            print("    -> The tail is rest (no load): the model reached the end of the test's "
                  "loaded portion; only the relaxation after it is missing.")

    sheet_result_df = pd.DataFrame({
        "Time [s]": t_eval_full,
        f"{sheet_name} Measured Voltage [V]": v_meas,
        "Simulated Voltage [V]": v_sim_interp,
        "Current [A]": i_meas,
        "Voltage Error [V]": residual,
        "Voltage Error [mV]": residual * 1000,
    })
    sheet_result_df.to_excel(excel_writer, sheet_name=sheet_name, index=False)

    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    max_ae = float(np.max(np.abs(residual)))
    ss_res = np.sum(residual ** 2)
    ss_tot = np.sum((v_meas - np.mean(v_meas)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    rest_mask = np.abs(i_meas) < 1e-3
    rest_rmse = float(np.sqrt(np.mean(residual[rest_mask] ** 2))) if rest_mask.any() else float("nan")
    rest_bias = float(np.mean(residual[rest_mask])) if rest_mask.any() else float("nan")
    coverage = float(t_eval_full[-1] / t_full[-1])

    all_metrics[sheet_name] = {
        "solver": solver_name,
        "coverage_of_profile": coverage,
        "coverage_of_loaded_profile": coverage_loaded,
        "sim_end_s": t_sim_end,
        "last_loaded_sample_s": t_load_end,
        "charge_required_Ah": q_required_ah,
        "model_deliverable_Ah": q_model_avail_ah,
        "rest_RMSE_mV": rest_rmse * 1000,
        "rest_bias_mV": rest_bias * 1000,
        "RMSE_mV": rmse * 1000,
        "MAE_mV": mae * 1000,
        "MaxAbsError_mV": max_ae * 1000,
        "R2": r2,
        "initial_soc": initial_soc,
        "solver_termination": str(solution.termination),
    }

    # 5.6 Plotting (1 figure per sheet)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    ax1.plot(t_eval_full / 3600, v_meas, label=f"Measured ({sheet_name})", color="tab:blue", lw=1.0)
    ax1.plot(t_eval_full / 3600, v_sim_interp, label="Simulated (DFN, Safari OCP)", color="tab:orange", lw=1.0, alpha=0.8)
    ax1.set_ylabel("Terminal Voltage [V]", fontweight="bold")
    ax1.set_title(f"Measured vs Simulated Voltage: {sheet_name}| RMSE: {rmse * 1000:.3f} mV | MAE: {mae * 1000:.3f} mV | R²: {r2:.5f}",fontsize=11, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(t_eval_full / 3600, residual * 1000, color="tab:red", lw=0.8)
    ax2.axhline(0, color="k", linestyle=":", lw=0.8)
    ax2.set_ylabel("Voltage Error [mV]", fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)

    ax3.plot(t_eval_full / 3600, i_meas, color="tab:green", lw=0.8)
    ax3.axhline(0, color="k", linestyle=":", lw=0.8)
    ax3.set_xlabel("Time [h]", fontweight="bold")
    ax3.set_ylabel("Current [A]", fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    img_path = os.path.join(OUT_DIR, f"fig_voltage_comparison_{sheet_name}.png")
    plt.savefig(img_path, dpi=200)
    plt.close(fig)

excel_writer.close()

# Save combined error metrics
with open(os.path.join(OUT_DIR, "all_sheets_error_metrics.json"), "w") as f:
    json.dump(all_metrics, f, indent=2)

print("\n" + "=" * 70)
print("ALL 5 SHEETS COMPLETED SUCCESSFULLY!")
print(f"Single Excel File Saved: {excel_output_path}")
print(f"5 Sheet Image Figures Saved to: {OUT_DIR}/")
print("=" * 70)