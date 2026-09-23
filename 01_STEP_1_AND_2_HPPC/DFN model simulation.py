import json
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_XLSX = INPUT_DIR / "All_MAT_Files_Converted.xlsx"
OUT_DIR = OUTPUT_DIR

# =============================================================================
# 1. DATA LOADING & CLEANING UTILITIES
# =============================================================================
def load_hppc_data(path: str = RAW_XLSX) -> pd.DataFrame:
    """Load and clean the HPPC sheet from the dataset."""
    df = pd.read_excel(path, sheet_name="HPPC")
    df = df.rename(
        columns={
            "Time": "t",
            "Voltage": "V",
            "Current": "I_raw",
            "Ah": "Ah",
            "Battery_Temp_degC": "T",
        }
    )

    # Sort by time and drop duplicate / non-increasing timestamps
    df = df.sort_values("t").reset_index(drop=True)
    keep = np.concatenate(([True], np.diff(df["t"].values) > 0))
    df = df.loc[keep].reset_index(drop=True)

    # Shift time to start at zero
    df["t"] = df["t"] - df["t"].iloc[0]

    # PyBaMM sign convention: + = discharge
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
# 2. MODEL DEFINITION & PARAMETERIZATION
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
    "Nominal cell capacity [A.h]",
    "Current function [A]",
    "Electrode height [m]",
    "Electrode width [m]",
    "Lower voltage cut-off [V]",
    "Upper voltage cut-off [V]",
]:
    print(f"  {k}: {parameter_values[k]}")
print("=" * 70)


# Scale electrode width to match 2.5 Ah capacity (2.5 / 2.3 scaling ratio)
capacity_scale = 2.5 / 2.3
parameter_values.update(
    {
        "Nominal cell capacity [A.h]": 2.5,
        "Electrode width [m]": (
            parameter_values["Electrode width [m]"] * capacity_scale
        ),
        "Lower voltage cut-off [V]": 2.0,
        "Upper voltage cut-off [V]": 3.8,
    }
)

initial_soc = 0.9649

parameter_values.set_initial_stoichiometries(
    initial_soc, param=model.param, known_value="cell capacity"
)

print(f'Capacity_scale: {capacity_scale:.4f}')
print("=" * 70)
print("Updated capacity, voltage cut-off & initial parameters:")
for k in [
    "Nominal cell capacity [A.h]",
    "Electrode width [m]",
    "Lower voltage cut-off [V]",
    "Upper voltage cut-off [V]",
    "Initial concentration in negative electrode [mol.m-3]",
    "Initial concentration in positive electrode [mol.m-3]",
]:
    if k == "Electrode width [m]":
        print(
            f"  Electrode width [m] ( x capacity_scale {capacity_scale:.4f}): "
            f"{parameter_values['Electrode width [m]']:.4f}"
        )
    else:
        print(f"  {k}: {parameter_values[k]}")
print(
    f"  Initial State of Charge (SoC): {initial_soc} ({initial_soc * 100:.2f}%)"
)
print("=" * 70)


# =============================================================================
# 3. LOAD DATA & PREPARE SOLVER INPUT
# =============================================================================
hppc = load_hppc_data()
hppc_solve = decimate_rests(hppc, rest_dt_target=5.0)

t_data = hppc_solve["t"].to_numpy()
i_data = hppc_solve["I_pybamm"].to_numpy()

current_interpolant = pybamm.Interpolant(
    t_data, i_data, pybamm.t, name="HPPC current interpolant"
)
parameter_values["Current function [A]"] = current_interpolant



# =============================================================================
# 4. SOLVE DFN MODEL
# =============================================================================
solver = pybamm.CasadiSolver(mode="safe", atol=1e-6, rtol=1e-6)
sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=solver)

print(f"Solving DFN model over full HPPC profile ({t_data[-1]:.0f} s / {t_data[-1]/3600:.2f} h)...")
t0 = time.time()
solution = sim.solve(t_eval=t_data)
print(f"Solve completed in {time.time() - t0:.1f} s")

# =============================================================================
# 5. POST-PROCESSING & ERROR METRICS
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

residual = v_sim_interp - v_meas  # simulated - measured

# =============================================================================
# EXPORT MEASURED VS SIMULATED RESULTS TO EXCEL
# =============================================================================

output_df = pd.DataFrame({
    "Time [s]": t_eval_full,
    "HPPC Voltage [V]": v_meas,
    "Simulated Voltage [V]": v_sim_interp,
    "Current [A]": i_meas,
    "Voltage Error [V]": residual,
    "Voltage Error [mV]": residual * 1000,
})

output_excel_path = os.path.join(
    OUT_DIR, "HPPC_DFN_Simulation_Results.xlsx"
)

output_df.to_excel(output_excel_path, index=False)

print(f"Output Excel file saved: {output_excel_path}")

rmse = float(np.sqrt(np.mean(residual ** 2)))
mae = float(np.mean(np.abs(residual)))
max_ae = float(np.max(np.abs(residual)))
ss_res = np.sum(residual ** 2)
ss_tot = np.sum((v_meas - np.mean(v_meas)) ** 2)
r2 = float(1 - ss_res / ss_tot)

metrics = {
    "RMSE_V": rmse,
    "RMSE_mV": rmse * 1000,
    "MAE_V": mae,
    "MAE_mV": mae * 1000,
    "MaxAbsError_V": max_ae,
    "MaxAbsError_mV": max_ae * 1000,
    "R2": r2,
    "n_points_scored": int(mask.sum()),
    "solver_termination": str(solution.termination),
}
with open(os.path.join(OUT_DIR, "task2_error_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("=" * 60)
print("ERROR METRICS (Baseline Prada2013):")
print(f"  RMSE        : {rmse * 1000:.3f} mV ({rmse:.5f} V)")
print(f"  MAE         : {mae * 1000:.3f} mV ({mae:.5f} V)")
print(f"  Max Error   : {max_ae * 1000:.3f} mV ({max_ae:.5f} V)")
print(f"  R^2         : {r2:.5f}")
print("=" * 60)

# =============================================================================
# 6. PLOTTING (Time kept in Hours: t_eval_full / 3600)
# =============================================================================

# PLOT 1: Voltage Comparison (Top), Voltage Error (Middle), Current Profile (Bottom)
fig1, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

# Top Subplot: Voltage Comparison
ax1.plot(t_eval_full / 3600, v_meas, label="Measured (HPPC)", color="tab:blue", lw=1.0)
ax1.plot(t_eval_full / 3600, v_sim_interp, label="Simulated (DFN, Prada2013)", color="tab:orange", lw=1.0, alpha=0.8)
ax1.set_ylabel("Terminal Voltage [V]", fontsize=11, fontweight="bold")
ax1.set_title(f"Measured vs Simulated HPPC Voltage| RMSE: {rmse * 1000:.3f} mV    |    MAE: {mae * 1000:.3f} mV    |    R²: {r2:.5f}", fontsize=11, fontweight="bold")
ax1.legend(loc="upper right", frameon=True)
ax1.grid(True, linestyle="--", alpha=0.5)

# Middle Subplot: Voltage Residual Error
ax2.plot(t_eval_full / 3600, residual * 1000, color="tab:red", lw=0.8, label="Voltage Error (Sim - Meas)")
ax2.axhline(0, color="k", linestyle=":", lw=0.8)
ax2.set_ylabel("Voltage Error [mV]", fontsize=11, fontweight="bold")
ax2.set_title("Voltage Error Plot", fontsize=12, fontweight="bold")
ax2.legend(loc="upper right", frameon=True)
ax2.grid(True, linestyle="--", alpha=0.5)

# Bottom Subplot: Applied Current vs Time
ax3.plot(t_eval_full / 3600, i_meas, color="tab:green", lw=0.8, label="Applied Current")
ax3.axhline(0, color="k", linestyle=":", lw=0.8)
ax3.set_xlabel("Time [h]", fontsize=11, fontweight="bold")
ax3.set_ylabel("Current [A]", fontsize=11, fontweight="bold")
ax3.set_title("Current Profile (- = discharge, + = charge)", fontsize=12, fontweight="bold")
ax3.legend(loc="upper right", frameon=True)
ax3.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_task2_voltage_comparison_and_error.png"), dpi=200)

# PLOT 2: Residuals vs Current (Time colorbar in Hours)
fig2, ax_res = plt.subplots(figsize=(8, 6))
sc = ax_res.scatter(i_meas, residual * 1000, c=t_eval_full / 3600, cmap="viridis", s=4, alpha=0.6)
ax_res.axhline(0, color="k", linestyle=":", lw=0.8)
ax_res.set_xlabel("Current [A] (- = discharge, + = charge)", fontsize=11, fontweight="bold")
ax_res.set_ylabel("Voltage Residual [mV] (Sim - Meas)", fontsize=11, fontweight="bold")
ax_res.set_title("Residuals vs Current", fontsize=12, fontweight="bold")
ax_res.grid(True, linestyle="--", alpha=0.5)

cbar = plt.colorbar(sc, ax=ax_res)
cbar.set_label("Time [h]", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_task2_residuals_vs_current.png"), dpi=200)

plt.show()