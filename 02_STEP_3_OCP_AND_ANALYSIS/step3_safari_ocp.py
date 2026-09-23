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
# -----------------------------------------------------------------------------

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


# Numpy versions of the same two functions, for the (non-symbolic) fitting
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
def load_hppc_data(path: str = RAW_XLSX) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="HPPC")
    df = df.rename(
        columns={"Time": "t", "Voltage": "V", "Current": "I_raw", "Ah": "Ah", "Battery_Temp_degC": "T"}
    )
    df = df.sort_values("t").reset_index(drop=True)
    keep = np.concatenate(([True], np.diff(df["t"].values) > 0))
    df = df.loc[keep].reset_index(drop=True)
    df["t"] = df["t"] - df["t"].iloc[0]
    df["I_pybamm"] = -df["I_raw"]
    return df[["t", "V", "I_raw", "I_pybamm", "Ah", "T"]]


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
# -----------------------------------------------------------------------------

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
# -----------------------------------------------------------------------------

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

p0 = [0.00, 0.80, 0.86, 0.03]  # paper's own Hypothesis-1-ish starting point
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
print(f"  (Paper's own Table II reference: x_min~0.00 x_max~0.80-0.90 "
      f"y_min~0.03 y_max~0.76-0.86)")
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
# 4b. CAPACITY MATCHING (NEW)
# -----------------------------------------------------------------------------

def window_capacity_ah(pv, electrode: str, sto_span: float) -> float:
    """Ah delivered by one electrode over a stoichiometry span."""
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
print(f"Negative particle diffusivity: 3e-15 -> {NEG_DIFFUSIVITY:.2e} m2/s")
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
# 5. INITIAL SOC BACKTRACKED FROM THE FIRST MEASURED HPPC VOLTAGE
# =============================================================================
hppc = load_hppc_data()
v_initial_hppc = float(hppc["V"].iloc[0])
print(f"First voltage measured in HPPC sheet: {v_initial_hppc:.4f} V")

soc_from_model_ocv = interp1d(check_ocv, check_soc, kind="linear", fill_value="extrapolate")
initial_soc = float(np.clip(soc_from_model_ocv(v_initial_hppc), 0.0, 1.0))
print(f"Backtracked Initial SoC from {v_initial_hppc:.4f} V: {initial_soc:.4f} ({initial_soc * 100:.2f}%)")

x_init = x_0 + initial_soc * (x_100 - x_0)
y_init = y_0 + initial_soc * (y_100 - y_0)
parameter_values.update(
    {
        "Initial concentration in negative electrode [mol.m-3]": (
            x_init * parameter_values["Maximum concentration in negative electrode [mol.m-3]"]
        ),
        "Initial concentration in positive electrode [mol.m-3]": (
            y_init * parameter_values["Maximum concentration in positive electrode [mol.m-3]"]
        ),
    }
)

print(f"Capacity_scale: {capacity_scale:.4f}")
print(f"  x_init={x_init:.4f}, y_init={y_init:.4f}")
print("=" * 70)

# =============================================================================
# 6. LOAD DATA & PREPARE SOLVER INPUT
# =============================================================================
hppc_solve = decimate_rests(hppc, rest_dt_target=5.0)
t_data = hppc_solve["t"].to_numpy()
i_data = hppc_solve["I_pybamm"].to_numpy()

current_interpolant = pybamm.Interpolant(
    t_data, i_data, pybamm.t, name=f"HPPC_current_interpolant_{RUN_TAG}"
)
parameter_values["Current function [A]"] = current_interpolant

# =============================================================================
# 7. SOLVE DFN MODEL
# =============================================================================

try:
    solver = pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)
    solver_name = "IDAKLUSolver"
except Exception as exc:  
    print(f"IDAKLU unavailable ({exc}); falling back to CasadiSolver(dt_max=10)")
    solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-6, dt_max=10)
    solver_name = "CasadiSolver(dt_max=10)"
print(f"Solver: {solver_name}")
sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver)

print(f"Solving DFN model over full HPPC profile ({t_data[-1]:.0f} s / {t_data[-1]/3600:.2f} h)...")
t0 = time.time()
solution = sim.solve(t_eval=t_data)
print(f"Solve completed in {time.time() - t0:.1f} s")
print(f"Solver termination reason: {solution.termination}")

# --- Sanity check 1: lithium conservation. Charge passed (integral of I dt)

_t = solution["Time [s]"].entries
_i = solution["Current [A]"].entries  # + = discharge
_q_applied_ah = float(np.trapezoid(_i, _t) / 3600.0) if hasattr(np, "trapezoid") else float(np.trapz(_i, _t) / 3600.0)
try:
    _li = solution["Total lithium in negative electrode [mol]"].entries
    _q_stored_ah = float(-(_li[-1] - _li[0]) * FARADAY / 3600.0)
    print(f"Conservation check: applied {_q_applied_ah:.4f} Ah vs "
          f"electrode lithium change {_q_stored_ah:.4f} Ah "
          f"(gap {abs(_q_applied_ah - _q_stored_ah) * 1000:.1f} mAh)")
    if abs(_q_applied_ah - _q_stored_ah) > 0.02:
        print("  WARNING: lithium is NOT conserved - solver is skipping current pulses.")
except KeyError:
    print("Conservation check skipped (variable not available in this PyBaMM version).")

t_end_sim = float(_t[-1])
t_end_data = float(t_data[-1])
if t_end_sim < t_end_data - 1.0:
    print(f"  WARNING: simulation stopped early at {t_end_sim / 3600:.2f} h of "
          f"{t_end_data / 3600:.2f} h ({solution.termination}). Metrics cover only that window.\n"
              "  Cause is usually graphite surface depletion in the 15C pulses: raise NEG_DIFFUSIVITY.")

# =============================================================================
# 8. POST-PROCESSING & ERROR METRICS
# =============================================================================
t_full = hppc["t"].to_numpy()
v_meas_full = hppc["V"].to_numpy()
sim_t = solution["Time [s]"].entries
sim_v = solution["Terminal voltage [V]"].entries

mask = t_full <= sim_t[-1]
v_sim_interp = np.interp(t_full[mask], sim_t, sim_v)
v_meas = v_meas_full[mask]
t_eval_full = t_full[mask]
i_meas = hppc["I_raw"].to_numpy()[mask]
residual = v_sim_interp - v_meas

output_df = pd.DataFrame({
    "Time [s]": t_eval_full, "HPPC Voltage [V]": v_meas,
    "Simulated Voltage [V]": v_sim_interp, "Current [A]": i_meas,
    "Voltage Error [V]": residual, "Voltage Error [mV]": residual * 1000,
})
output_df.to_excel(os.path.join(OUT_DIR, "HPPC_DFN_Simulation_Results_SafariOCP.xlsx"), index=False)

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

metrics = {
    "solver": solver_name,
    "coverage_of_profile": coverage,
    "rest_RMSE_mV": rest_rmse * 1000,
    "rest_bias_mV": rest_bias * 1000,
    "capacity_matching": {
        "target_Ah": TARGET_CAPACITY_AH,
        "eps_n": eps_n_new, "eps_p": eps_p_new,
        "neg_diffusivity_m2_s": NEG_DIFFUSIVITY,
        "Q_n_before_Ah": Q_n_before, "Q_p_before_Ah": Q_p_before,
    },
    "RMSE_mV": rmse * 1000, "MAE_mV": mae * 1000, "MaxAbsError_mV": max_ae * 1000, "R2": r2,
    "n_points_scored": int(mask.sum()),
    "solver_termination": str(solution.termination),
    "fitted_stoichiometry_window": {"x_0": x_0, "x_100": x_100, "y_0": y_0, "y_100": y_100},
    "ocv_fit_rmse_mV": fit_rmse * 1000,
    "initial_soc": initial_soc,
}
with open(os.path.join(OUT_DIR, "task3_error_metrics_safari_ocp.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("=" * 60)
print("ERROR METRICS (Safari & Delacourt 2011 OCPs + fitted stoichiometry window):")
print(f"  RMSE        : {rmse * 1000:.3f} mV ({rmse:.5f} V)")
print(f"  MAE         : {mae * 1000:.3f} mV ({mae:.5f} V)")
print(f"  Max Error   : {max_ae * 1000:.3f} mV ({max_ae:.5f} V)")
print(f"  R^2         : {r2:.5f}")
print(f"  Rest-only   : RMSE {rest_rmse * 1000:.2f} mV, bias {rest_bias * 1000:+.2f} mV")
print(f"  Profile covered by simulation: {coverage * 100:.1f} %")
print("=" * 60)

# =============================================================================
# 9. PLOTTING
# =============================================================================
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
ax1.plot(t_eval_full / 3600, v_meas, label="Measured (HPPC)", color="tab:blue", lw=1.0)
ax1.plot(t_eval_full / 3600, v_sim_interp, label="Simulated (DFN, Safari OCP)", color="tab:orange", lw=1.0, alpha=0.8)
ax1.set_ylabel("Terminal Voltage [V]", fontweight="bold")
ax1.set_title(f"Measured vs Simulated HPPC Voltage (Safari & Delacourt OCPs)| RMSE: {rmse * 1000:.3f} mV | MAE: {mae * 1000:.3f} mV | R²: {r2:.5f}",fontsize=11, fontweight="bold")
ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.5)

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
plt.savefig(os.path.join(OUT_DIR, "fig_task3_With-OCP-Function_safari_voltage_comparison_and_error.png"), dpi=200)

fig2, ax_res = plt.subplots(figsize=(8, 6))
sc = ax_res.scatter(i_meas, residual * 1000, c=t_eval_full / 3600, cmap="viridis", s=4, alpha=0.6)
ax_res.axhline(0, color="k", linestyle=":", lw=0.8)
ax_res.set_xlabel("Current [A] (- = discharge, + = charge)", fontweight="bold")
ax_res.set_ylabel("Voltage Residual [mV] (Sim - Meas)", fontweight="bold")
ax_res.set_title("Residuals vs Current", fontweight="bold")
ax_res.grid(True, linestyle="--", alpha=0.5)
cbar = plt.colorbar(sc, ax=ax_res); cbar.set_label("Time [h]")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_task3_safari_residuals_vs_current.png"), dpi=200)

fig3, ax_ocv = plt.subplots(figsize=(7, 5))
ax_ocv.plot(soc_meas * 100, ocv_meas, label="Measured OCV (25C)", lw=1.5)
ax_ocv.plot(check_soc * 100, check_ocv, label="Fitted model OCV (Safari & Delacourt Up/Un)", lw=1.5, ls="--")
ax_ocv.axvline(initial_soc * 100, color="gray", ls=":", lw=1, label=f"Initial SOC = {initial_soc*100:.1f}%")
ax_ocv.set_xlabel("SOC [%]"); ax_ocv.set_ylabel("OCV [V]")
ax_ocv.set_title(f"Safari & Delacourt OCP fit (RMSE = {fit_rmse*1000:.2f} mV)")
ax_ocv.legend(); ax_ocv.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_task3_safari_ocv_fit_check.png"), dpi=200)

# Individual electrode OCP curves - sanity check plot against Fig. 2a/2b of the paper
fig4, (axp, axn) = plt.subplots(1, 2, figsize=(11, 4.5))
y_range = np.linspace(0.001, 0.999, 500)
x_range = np.linspace(0.001, 0.999, 500)
axp.plot(y_range, _up_np(y_range), color="tab:red")
axp.set_xlabel("y (LFP stoichiometry)"); axp.set_ylabel("U_p [V vs Li]")
axp.set_title("Cathode OCP"); axp.grid(True, linestyle="--", alpha=0.5)
axn.plot(x_range, _un_np(x_range), color="tab:purple")
axn.set_xlabel("x (graphite stoichiometry)"); axn.set_ylabel("U_n [V vs Li]")
axn.set_title("Anode OCP"); axn.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_task3_safari_electrode_ocp_sanity_check.png"), dpi=200)

plt.show()  