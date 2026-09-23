import os
import json
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "inputs"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_XLSX = INPUT_DIR / "All_MAT_Files_Converted.xlsx"
OCV_XLSX = INPUT_DIR / "OCV_25C_Clean_Monotonic.xlsx"
OUT_DIR = OUTPUT_DIR
PARAMS_JSON = os.path.join(OUT_DIR, "optimized_parameters.json")
RUN_TAG = f"{int(time.time() * 1000)}"

if not os.path.exists(PARAMS_JSON):
    raise SystemExit(f"{PARAMS_JSON} not found - run step4_optimize.py first.")
with open(PARAMS_JSON) as fh:
    OPT = json.load(fh)

TARGET_CAPACITY_AH = OPT["fixed_from_step3"]["target_capacity_Ah"]
NEG_DIFFUSIVITY = OPT["fixed_from_step3"]["neg_diffusivity_baseline_m2_s"]
FARADAY = 96485.33212

TRAIN_SHEETS = OPT["training_sheets"]
ALL_SHEETS = TRAIN_SHEETS + ["HPPC"]   # HPPC is the Step 5 validation set (never used to fit the parameters above)
VOLT = "Terminal voltage [V]"


# =============================================================================
# 1. SAFARI & DELACOURT (2011) OCP FUNCTIONS - identical to Step 3 / Step 4
# =============================================================================
def positive_electrode_ocp_safari(sto):
    y = sto
    return (
        3.4323 - 0.8428 * pybamm.exp(-80.2493 * (1 - y) ** 1.3198)
        - 3.2474e-6 * pybamm.exp(20.2645 * (1 - y) ** 3.8003)
        + 3.2482e-6 * pybamm.exp(20.2646 * (1 - y) ** 3.7995)
    )


def negative_electrode_ocp_safari(sto):
    x = sto
    return (
        0.6379 + 0.5416 * pybamm.exp(-305.5309 * x)
        + 0.044 * pybamm.tanh(-(x - 0.1958) / 0.1088)
        - 0.1978 * pybamm.tanh((x - 1.0571) / 0.0854)
        - 0.6875 * pybamm.tanh((x + 0.0117) / 0.0529)
        - 0.0175 * pybamm.tanh((x - 0.5692) / 0.0875)
    )


def _up_np(y):
    return (
        3.4323 - 0.8428 * np.exp(-80.2493 * (1 - y) ** 1.3198)
        - 3.2474e-6 * np.exp(20.2645 * (1 - y) ** 3.8003)
        + 3.2482e-6 * np.exp(20.2646 * (1 - y) ** 3.7995)
    )


def _un_np(x):
    return (
        0.6379 + 0.5416 * np.exp(-305.5309 * x)
        + 0.044 * np.tanh(-(x - 0.1958) / 0.1088)
        - 0.1978 * np.tanh((x - 1.0571) / 0.0854)
        - 0.6875 * np.tanh((x + 0.0117) / 0.0529)
        - 0.0175 * np.tanh((x - 0.5692) / 0.0875)
    )


# =============================================================================
# 2. DATA LOADING (identical to Step 3 / Step 4)
# =============================================================================
def load_sheet_data(sheet_name, path=RAW_XLSX):
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.rename(columns={"Time": "t", "Voltage": "V", "Current": "I_raw"})
    df = df.sort_values("t").reset_index(drop=True)
    keep = np.concatenate(([True], np.diff(df["t"].values) > 0))
    df = df.loc[keep].reset_index(drop=True)
    df["t"] = df["t"] - df["t"].iloc[0]
    df["I_pybamm"] = -df["I_raw"]
    return df[["t", "V", "I_raw", "I_pybamm"]]


def load_hppc_data(path=RAW_XLSX):
    df = pd.read_excel(path, sheet_name="HPPC")
    df = df.rename(columns={"Time": "t", "Voltage": "V", "Current": "I_raw"})
    df = df.sort_values("t").reset_index(drop=True)
    keep = np.concatenate(([True], np.diff(df["t"].values) > 0))
    df = df.loc[keep].reset_index(drop=True)
    df["t"] = df["t"] - df["t"].iloc[0]
    df["I_pybamm"] = -df["I_raw"]
    return df[["t", "V", "I_raw", "I_pybamm"]]


def decimate_rests(df, rest_dt_target=5.0, current_tol=1e-3):
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


def load_sheet(name):
    return load_hppc_data() if name == "HPPC" else load_sheet_data(name)


# =============================================================================
# 3. MODEL, BASE PARAMETERIZATION, STOICHIOMETRY WINDOW, CAPACITY MATCHING
#    (identical to Step 3 / Step 4)
# =============================================================================
model = pybamm.lithium_ion.DFN(
    options={"thermal": "isothermal", "surface form": "false", "particle": "Fickian diffusion",
             "contact resistance": "true"}
)

parameter_values = pybamm.ParameterValues("Prada2013")
capacity_scale = 2.5 / 2.3
parameter_values.update({
    "Nominal cell capacity [A.h]": 2.5,
    "Electrode width [m]": parameter_values["Electrode width [m]"] * capacity_scale,
    "Lower voltage cut-off [V]": 2.0,
    "Upper voltage cut-off [V]": 3.8,
})
parameter_values["Negative electrode OCP [V]"] = negative_electrode_ocp_safari
parameter_values["Positive electrode OCP [V]"] = positive_electrode_ocp_safari

x_0 = OPT["fixed_from_step3"]["stoich_window"]["x_0"]
x_100 = OPT["fixed_from_step3"]["stoich_window"]["x_100"]
y_0 = OPT["fixed_from_step3"]["stoich_window"]["y_0"]
y_100 = OPT["fixed_from_step3"]["stoich_window"]["y_100"]
check_soc = np.linspace(0, 1, 1001)
check_ocv_base = _up_np(y_0 + check_soc * (y_100 - y_0)) - _un_np(x_0 + check_soc * (x_100 - x_0))
soc_from_model_ocv = interp1d(check_ocv_base, check_soc, kind="linear", fill_value="extrapolate")

parameter_values["Negative particle diffusivity [m2.s-1]"] = NEG_DIFFUSIVITY
parameter_values.update({
    "Negative electrode active material volume fraction": OPT["fixed_from_step3"]["eps_n"],
    "Positive electrode active material volume fraction": OPT["fixed_from_step3"]["eps_p"],
})

print("=" * 70)
print(f"Stoichiometry window (fixed from Step 4): x_0={x_0:.4f} x_100={x_100:.4f} y_0={y_0:.4f} y_100={y_100:.4f}")
print(f"Capacity matched to {TARGET_CAPACITY_AH} Ah | Negative particle diffusivity baseline {NEG_DIFFUSIVITY:.2e} m2/s")
print("=" * 70)


# =============================================================================
# 4. APPLY THE OPTIMIZED PARAMETERS
# =============================================================================
opt = OPT["optimized"]


def _wrap(base_func, multiplier):
    return lambda *a, **k: base_func(*a, **k) * multiplier


scalar_updates = {}
function_updates = {}
for name, info in opt.items():
    if name in ("R_contact_ohm", "ocv_correction_mV"):
        continue
    key = info["parameter"]
    if info["is_function"]:
        function_updates[key] = _wrap(parameter_values[key], info["multiplier"])
    else:
        scalar_updates[key] = info["absolute_value"]

parameter_values.update(scalar_updates)          # plain parameter_values.update() for every scalar
parameter_values.update(function_updates)        # parameter_values.update() with a wrapper, for function-type parameters only
parameter_values.update({"Contact resistance [Ohm]": opt["R_contact_ohm"]})

C = opt["ocv_correction_mV"]
END_WIDTH = 0.03

def positive_ocp_optimized(sto):
    """Fixed (no InputParameter) version of the Step 4 OCV correction, baked in with the optimized coefficients."""
    s = (sto - y_0) / (y_100 - y_0)
    dU = (C["ocv_c0"] + C["ocv_c1"] * (1 - s) + C["ocv_c2"] * (1 - s) ** 2
          + C["ocv_end_lo"] * pybamm.exp(-s / END_WIDTH) + C["ocv_end_hi"] * pybamm.exp(-(1 - s) / END_WIDTH)) / 1000.0
    return positive_electrode_ocp_safari(sto) + dU

parameter_values.update({"Positive electrode OCP [V]": positive_ocp_optimized})   # necessary exception (function parameter)

# sanity: report what the corrected OCV looks like and confirm it is monotonic
dU_np = (C["ocv_c0"] + C["ocv_c1"] * (1 - check_soc) + C["ocv_c2"] * (1 - check_soc) ** 2
         + C["ocv_end_lo"] * np.exp(-check_soc / END_WIDTH) + C["ocv_end_hi"] * np.exp(-(1 - check_soc) / END_WIDTH)) / 1000.0
check_ocv_opt = check_ocv_base + dU_np
soc_from_optimized_ocv = interp1d(check_ocv_opt, check_soc, kind="linear", fill_value="extrapolate")
print("Applied optimized parameters:")
for n, info in opt.items():
    if n in ("R_contact_ohm", "ocv_correction_mV"):
        continue
    tag = "(function)" if info.get("is_function") else f"= {info.get('absolute_value'):.4e}"
    print(f"   {n:8s} x{info['multiplier']:.4f} {tag}")
print(f"   R_contact = {opt['R_contact_ohm'] * 1000:.3f} mOhm")
print(f"   OCV correction (mV): {C}")
print(f"   Corrected OCV monotonic: {'yes' if np.all(np.diff(check_ocv_opt) > 0) else 'NO - USE step4_optimize.py --reset AND RE-OPTIMIZE'}")
if OPT.get("at_bounds_treat_with_caution"):
    print(f"   NOTE - not identified from the training data (at their bounds): {OPT['at_bounds_treat_with_caution']}")
print("=" * 70)

# =============================================================================
# 5. SIMULATE EVERY SHEET WITH THE OPTIMIZED PARAMETERS, SCORE, PLOT
# =============================================================================
PHYS_VARS = [
    "Negative electrode surface potential difference at separator interface [V]",
    "X-averaged negative electrode surface potential difference [V]",
    "X-averaged positive electrode surface potential difference [V]",
    "X-averaged negative particle surface stoichiometry",
    "X-averaged positive particle surface stoichiometry",
]
BASE_VARS = [VOLT, "Current [A]", "Total lithium in negative electrode [mol]"]

def make_solver(out_vars):
    for opts_ in ({"num_threads": 1}, None):
        try:
            kw = {"output_variables": out_vars}
            if opts_ is not None:
                kw["options"] = opts_
            return pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6, **kw)
        except TypeError:
            continue
    return pybamm.IDAKLUSolver(atol=1e-6, rtol=1e-6)

def step3_metrics(t_full, v_meas_full, i_raw_full, t_sim, v_sim):
    mask = t_full <= t_sim[-1]
    v_i = np.interp(t_full[mask], t_sim, v_sim)
    res = v_i - v_meas_full[mask]
    ss_tot = np.sum((v_meas_full[mask] - np.mean(v_meas_full[mask])) ** 2)
    rest = np.abs(i_raw_full[mask]) < 1e-3
    return {
        "RMSE_mV": float(np.sqrt(np.mean(res ** 2)) * 1000), "MAE_mV": float(np.mean(np.abs(res)) * 1000),
        "MaxAbsError_mV": float(np.max(np.abs(res)) * 1000),
        "R2": float(1 - np.sum(res ** 2) / ss_tot) if ss_tot > 0 else float("nan"),
        "rest_RMSE_mV": float(np.sqrt(np.mean(res[rest] ** 2)) * 1000) if rest.any() else float("nan"),
        "rest_bias_mV": float(np.mean(res[rest]) * 1000) if rest.any() else float("nan"),
        "coverage_of_profile": float(t_full[mask][-1] / t_full[-1]),
    }

def _extras(sol, names):
    out = {}
    for nm in names:
        try:
            out[nm] = np.asarray(sol[nm].entries, dtype=float)
        except Exception:
            pass
    return out

def run_sheet(name, extra=()):
    df = load_sheet(name)
    v0 = float(df["V"].iloc[0])
    soc0 = float(np.clip(soc_from_optimized_ocv(v0), 0.0, 1.0))
    x_init = x_0 + soc0 * (x_100 - x_0); y_init = y_0 + soc0 * (y_100 - y_0)
    hs = decimate_rests(df, rest_dt_target=5.0)
    t_data, i_data = hs["t"].to_numpy(), hs["I_pybamm"].to_numpy()
    pv = parameter_values.copy()
    pv.update({
        "Initial concentration in negative electrode [mol.m-3]": x_init * pv["Maximum concentration in negative electrode [mol.m-3]"],
        "Initial concentration in positive electrode [mol.m-3]": y_init * pv["Maximum concentration in positive electrode [mol.m-3]"],
        "Current function [A]": pybamm.Interpolant(t_data, i_data, pybamm.t, name=f"cur_{name}_{RUN_TAG}"),
    })
    vars_ = BASE_VARS + list(extra)
    sim = pybamm.Simulation(model, parameter_values=pv, solver=make_solver(vars_))
    try:
        sol = sim.solve(t_eval=t_data)
    except pybamm.SolverError:
        sol = sim.solution
    ts, vs = sol["Time [s]"].entries, sol[VOLT].entries
    m = step3_metrics(df["t"].to_numpy(), df["V"].to_numpy(), df["I_raw"].to_numpy(), ts, vs)
    m["termination"] = str(sol.termination); m["initial_soc"] = soc0
    try:
        _i = sol["Current [A]"].entries
        q_app = float((np.trapezoid(_i, ts) if hasattr(np, "trapezoid") else np.trapz(_i, ts)) / 3600.0)
        li = sol["Total lithium in negative electrode [mol]"].entries
        m["conservation_gap_mAh"] = abs(q_app + (li[-1] - li[0]) * FARADAY / 3600.0) * 1000
    except Exception:
        pass
    if name == "HPPC":
        Ia = np.abs(df["I_raw"].to_numpy()); msk = df["t"].to_numpy() <= ts[-1]
        res = np.interp(df["t"].to_numpy()[msk], ts, vs) - df["V"].to_numpy()[msk]
        for lab, sel in {"RMSE_pulses_le_10A_mV": (Ia >= 1e-3) & (Ia <= 10), "RMSE_pulses_gt_10A_mV": Ia > 10}.items():
            s = sel[msk]
            if s.any():
                m[lab] = float(np.sqrt(np.mean(res[s] ** 2)) * 1000)
    ex = _extras(sol, extra); ex["_t"] = np.asarray(ts, dtype=float)
    return df, ts, vs, m, ex

def plot_physical(title, t_s, ex, path):
    a_sep = ex.get("Negative electrode surface potential difference at separator interface [V]")
    a_avg = ex.get("X-averaged negative electrode surface potential difference [V]")
    c_avg = ex.get("X-averaged positive electrode surface potential difference [V]")
    x_s = ex.get("X-averaged negative particle surface stoichiometry")
    y_s = ex.get("X-averaged positive particle surface stoichiometry")
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    if a_sep is not None:
        ax[0].plot(t_s / 3600, a_sep, lw=0.8, label="anode potential at separator interface (vs Li/Li+)")
    if a_avg is not None:
        ax[0].plot(t_s / 3600, a_avg, lw=0.8, label="anode potential, x-averaged (vs Li/Li+)")
    ax[0].axhline(0, color="r", ls="--", lw=0.9, label="0 V = lithium-plating threshold")
    ax[0].set_ylabel("Anode potential [V vs Li/Li+]"); ax[0].legend(fontsize=8); ax[0].set_title(title, fontweight="bold")
    if c_avg is not None:
        ax[1].plot(t_s / 3600, c_avg, color="tab:red", lw=0.8, label="cathode potential, x-averaged (vs Li/Li+)")
        ax[1].set_ylabel("Cathode potential [V vs Li/Li+]"); ax[1].legend(fontsize=8)
    if x_s is not None:
        ax[2].plot(t_s / 3600, x_s, lw=0.8, label="graphite surface stoichiometry")
    if y_s is not None:
        ax[2].plot(t_s / 3600, y_s, lw=0.8, label="LFP surface stoichiometry")
    ax[2].set_ylabel("Surface stoichiometry [-]"); ax[2].set_xlabel("Time [h]"); ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.4, ls="--")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close(fig)

def main():
    print("\n" + "=" * 78 + "\nSTEP 5: VALIDATION with the optimized parameters\n" + "=" * 78)
    results, curves = {}, {}
    with pd.ExcelWriter(os.path.join(OUT_DIR, "Validation_Results.xlsx"), engine="openpyxl") as xw:
        for name in TRAIN_SHEETS:
            extra = PHYS_VARS if name in ("Charge_4C", "Discharge_1C") else ()
            df, ts, vs, m, ex = run_sheet(name, extra=extra)
            results[name] = m
            vsim = np.interp(df["t"].to_numpy(), ts, vs); vsim[df["t"].to_numpy() > ts[-1]] = np.nan
            print(f"{name:13s} RMSE {m['RMSE_mV']:7.2f} mV | rest RMSE {m['rest_RMSE_mV']:6.2f} (bias {m['rest_bias_mV']:+.1f}) | "
                  f"coverage {m['coverage_of_profile'] * 100:5.1f} % | {m['termination']}")
            pd.DataFrame({"Time [s]": df["t"], "Measured Voltage [V]": df["V"], "Optimized Simulated [V]": vsim,
                          "Current [A]": df["I_raw"], "Error [mV]": (vsim - df["V"]) * 1000}
                         ).to_excel(xw, sheet_name=name, index=False)
            fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            a1.plot(df["t"], df["V"], "b", lw=1.0, label="Measured")
            a1.plot(df["t"], vsim, color="tab:green", lw=0.9, label="Optimized model")
            a1.set_ylabel("Voltage [V]"); a1.legend(); a1.grid(alpha=.4)
            a1.set_title(f"{name}: RMSE {m['RMSE_mV']:.1f} mV | MAE {m['MAE_mV']:.1f} mV | R2 {m['R2']:.5f}", fontsize=11, fontweight="bold")
            a2.plot(df["t"], (vsim - df["V"]) * 1000, color="tab:green", lw=0.8); a2.axhline(0, color="k", lw=.6)
            a2.set_ylabel("Error [mV]"); a2.set_xlabel("Time [s]"); a2.grid(alpha=.4)
            plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"fig_val_{name}.png"), dpi=150); plt.close(fig)
            if extra:
                plot_physical(f"Optimized model - internal variables, {name}", ex["_t"], ex, os.path.join(OUT_DIR, f"fig_val_physical_{name}.png"))

        print("\n" + "-" * 78 + "\nHPPC (validation - never used to fit the parameters above)\n" + "-" * 78)
        df, ts, vs, m, ex = run_sheet("HPPC", extra=PHYS_VARS)
        results["HPPC"] = m
        vsim = np.interp(df["t"].to_numpy(), ts, vs); vsim[df["t"].to_numpy() > ts[-1]] = np.nan
        print(f"{'HPPC':13s} RMSE {m['RMSE_mV']:7.2f} mV | rest RMSE {m['rest_RMSE_mV']:6.2f} (bias {m['rest_bias_mV']:+.1f}) | "
              f"<=10A pulses {m.get('RMSE_pulses_le_10A_mV', float('nan')):.1f} | >10A pulses {m.get('RMSE_pulses_gt_10A_mV', float('nan')):.1f} | "
              f"coverage {m['coverage_of_profile'] * 100:.1f} % | {m['termination']}"
              + (f" | Li conservation gap {m['conservation_gap_mAh']:.1f} mAh" if "conservation_gap_mAh" in m else ""))
        pd.DataFrame({"Time [s]": df["t"], "Measured Voltage [V]": df["V"], "Optimized Simulated [V]": vsim,
                      "Current [A]": df["I_raw"], "Error [mV]": (vsim - df["V"]) * 1000}
                     ).to_excel(xw, sheet_name="HPPC_validation", index=False)
        fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
        a1.plot(df["t"] / 3600, df["V"], "b", lw=0.9, label="Measured")
        a1.plot(df["t"] / 3600, vsim, color="tab:green", lw=0.8, label="Optimized model")
        a2.plot(df["t"] / 3600, (vsim - df["V"]) * 1000, color="tab:green", lw=0.7)
        a3.plot(df["t"] / 3600, df["I_raw"], "g", lw=0.7)
        a1.set_ylabel("Voltage [V]"); a1.legend()
        a1.set_title(f"HPPC validation: RMSE {m['RMSE_mV']:.1f} mV | MAE {m['MAE_mV']:.1f} mV | R2 {m['R2']:.5f}", fontsize=11, fontweight="bold")
        a2.axhline(0, color="k", lw=.6); a2.set_ylabel("Error [mV]"); a3.set_ylabel("Current [A]"); a3.set_xlabel("Time [h]")
        for a in (a1, a2, a3):
            a.grid(alpha=.4, ls="--")
        plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_val_hppc.png"), dpi=170); plt.close(fig)
        plot_physical("Optimized model - internal variables, HPPC validation", ex["_t"], ex, os.path.join(OUT_DIR, "fig_val_physical_hppc.png"))

        pd.DataFrame([{"Dataset": k, **v} for k, v in results.items()]).to_excel(xw, sheet_name="Summary", index=False)

    with open(os.path.join(OUT_DIR, "validation_metrics.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)

    print("\n" + "=" * 78)
    print("Why RMSE (root-mean-square error) is used to score simulated vs measured voltage:")
    print(" - Same units (V) as the quantity itself, so the number is directly interpretable.")
    print(" - Squares the residual before averaging, so it penalises large, isolated deviations")
    print("   (e.g. a missed pulse peak) more than a metric like MAE would - appropriate here")
    print("   because a large one-off voltage error is more diagnostic of a wrong physical")
    print("   parameter than many small ones.")
    print("   MAE and R^2 are reported alongside it for context; MaxAbsError flags any single worst point.")
    print("=" * 78)
    print(f"\nSaved to {os.path.abspath(OUT_DIR)}: validation_metrics.json, Validation_Results.xlsx, fig_val_*.png")


if __name__ == "__main__":
    main()