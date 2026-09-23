import os
import sys
import builtins
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_real_print = builtins.print
if any(a in sys.argv for a in ("--worker",)):
    builtins.print = lambda *a, **k: None     
import matplotlib
matplotlib.use("Agg")
import shutil
import json
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
# 1. SAFARI & DELACOURT (2011) OCP FUNCTIONS 
# =============================================================================
def positive_electrode_ocp_safari(sto):
    y = sto
    return (
        3.4323
        - 0.8428 * pybamm.exp(-80.2493 * (1 - y) ** 1.3198)
        - 3.2474e-6 * pybamm.exp(20.2645 * (1 - y) ** 3.8003)
        + 3.2482e-6 * pybamm.exp(20.2646 * (1 - y) ** 3.7995)
    )

def negative_electrode_ocp_safari(sto):
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
# 2. DATA LOADING & CLEANING UTILITIES (identical to Step 3)
# =============================================================================
def load_sheet_data(sheet_name: str, path: str = RAW_XLSX) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.rename(columns={"Time": "t", "Voltage": "V", "Current": "I_raw"})
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
# 3. MODEL DEFINITION & BASE PARAMETERIZATION (identical to Step 3)
# =============================================================================
model = pybamm.lithium_ion.DFN(
    options={"thermal": "isothermal", "surface form": "false", "particle": "Fickian diffusion"}
)

parameter_values = pybamm.ParameterValues("Prada2013")

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
# 4. STOICHIOMETRY WINDOW (fitted once to the measured OCV - identical to Step 3)
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


result = least_squares(
    residuals, [0.00, 0.80, 0.86, 0.03], args=(soc_meas, ocv_meas),
    bounds=([0.00, 0.65, 0.55, 0.00], [0.20, 0.95, 0.95, 0.30])
)
x_0, x_100, y_0, y_100 = result.x
check_soc = np.linspace(0, 1, 1001)
check_ocv = model_ocv_np(result.x, check_soc)
if np.any(np.diff(check_ocv) < 0):
    raise RuntimeError("Fitted model OCV is non-monotonic - abort.")
soc_from_model_ocv = interp1d(check_ocv, check_soc, kind="linear", fill_value="extrapolate")

print("=" * 70)
print(f"Stoichiometry window: x_0={x_0:.4f} x_100={x_100:.4f} y_0={y_0:.4f} y_100={y_100:.4f}")
print("=" * 70)

# =============================================================================
# 4b. CAPACITY MATCHING (identical to Step 3)
# =============================================================================
def window_capacity_ah(pv, electrode, sto_span):
    area = pv["Electrode height [m]"] * pv["Electrode width [m]"]
    eps = pv[f"{electrode} electrode active material volume fraction"]
    thick = pv[f"{electrode} electrode thickness [m]"]
    c_max = pv[f"Maximum concentration in {electrode.lower()} electrode [mol.m-3]"]
    return eps * thick * area * c_max * FARADAY / 3600.0 * sto_span


span_n, span_p = x_100 - x_0, y_0 - y_100
Q_n_before = window_capacity_ah(parameter_values, "Negative", span_n)
Q_p_before = window_capacity_ah(parameter_values, "Positive", span_p)
eps_n_old = parameter_values["Negative electrode active material volume fraction"]
eps_p_old = parameter_values["Positive electrode active material volume fraction"]
eps_n_new = eps_n_old * TARGET_CAPACITY_AH / Q_n_before
eps_p_new = eps_p_old * TARGET_CAPACITY_AH / Q_p_before
parameter_values["Negative particle diffusivity [m2.s-1]"] = NEG_DIFFUSIVITY
parameter_values.update({
    "Negative electrode active material volume fraction": eps_n_new,
    "Positive electrode active material volume fraction": eps_p_new,
})
print(f"Capacity matched to {TARGET_CAPACITY_AH} Ah | eps_n {eps_n_old:.4f}->{eps_n_new:.4f} | "
      f"eps_p {eps_p_old:.4f}->{eps_p_new:.4f} | Negative particle diffusivity -> {NEG_DIFFUSIVITY:.2e} m2/s")
print("=" * 70)


# =============================================================================
# ==========================  STEP 4 : OPTIMIZATION  =========================
#   python step4_optimize.py                    -> run (or resume) the optimization
#   python step4_optimize.py --reset             -> delete saved progress, start over
#   options: --chunk-seconds 600  --max-evals 90  --mem-limit-mb 3000
# =============================================================================
import argparse
import gc
import hashlib
import subprocess
try:
    import psutil
except Exception:
    psutil = None

SHEETS = ["Charge_1C", "Charge_2C", "Charge_3C", "Charge_4C", "Discharge_1C"] 
N_EVAL_PTS = 1500
N_SOLVE_PTS = 400
N_LHS = 8
MAX_TRF_NFEV = 30
FREEZE_FRACTION = 0.03
R_UNIT = 5e-3
DEFAULT_CHUNK_SECONDS = 600
DEFAULT_MAX_EVALS = 90
DEFAULT_MEM_LIMIT_MB = 3000
MAX_EVALS_PER_CHUNK = 40
TAIL_CAP_V = 0.10        # if a sheet's simulation stops early, missing samples count at most this much
FIT_ATOL, FIT_RTOL = 1e-4, 1e-3   # loosened tolerance DURING the fit only - see note above
END_WIDTH = 0.03          # width (in SOC) of the two end-of-range OCV correction bumps
ALIGN_MIN_REST_S = 300    # rests at least this long (in the training sheets) seed the OCV-correction start guess
ALIGN_RIDGE = 0.3
ALIGN_LIMITS_MV = np.array([60.0, 150.0, 150.0, 150.0, 150.0])   # bounds of c0, c1, c2, e_lo, e_hi
JAC_REFRESH_EVERY = 8

VOLT = "Terminal voltage [V]"

STATE_DIR = os.path.join(OUT_DIR, "step4_state")
EVAL_DIR = os.path.join(STATE_DIR, "evals")
META_JSON = os.path.join(STATE_DIR, "meta.json")
RESULT_JSON = os.path.join(STATE_DIR, "optimization_result.json")
EVAL_INDEX = os.path.join(STATE_DIR, "evals.jsonl")
ALIGN_JSON = os.path.join(STATE_DIR, "ocv_start.json")
FINAL_JSON = os.path.join(OUT_DIR, "optimized_parameters.json")


class BudgetExhausted(Exception):
    pass

class CapReached(Exception):
    pass

def lower_priority():
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
        else:
            os.nice(10)
    except Exception:
        pass

def rss_mb():
    if psutil is None:
        return 0.0
    try:
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0

def make_solver(out_vars, fit=True):
    atol, rtol = (FIT_ATOL, FIT_RTOL) if fit else (1e-6, 1e-6)
    try:
        return pybamm.IDAKLUSolver(atol=atol, rtol=rtol, output_variables=list(dict.fromkeys(out_vars)),
                                    options={"num_threads": 1})
    except TypeError:
        print("WARNING: this PyBaMM version cannot restrict stored variables; consider `pip install -U pybamm`.")
        return pybamm.IDAKLUSolver(atol=atol, rtol=rtol)

model_opt = pybamm.lithium_ion.DFN(
    options={"thermal": "isothermal", "surface form": "false", "particle": "Fickian diffusion",
             "contact resistance": "true"}
)

# ---- candidate parameters --------------------------------------------------
CANDIDATES = [
    ("k_n", "Negative electrode exchange-current density [A.m-2]"),
    ("k_p", "Positive electrode exchange-current density [A.m-2]"),
    ("D_n", "Negative particle diffusivity [m2.s-1]"),
    ("D_p", "Positive particle diffusivity [m2.s-1]"),
    ("sig_n", "Negative electrode conductivity [S.m-1]"),
    ("sig_p", "Positive electrode conductivity [S.m-1]"),
    ("kappa_e", "Electrolyte conductivity [S.m-1]"),
    ("D_e", "Electrolyte diffusivity [m2.s-1]"),
]

def _has(k):
    try:
        parameter_values[k]
        return True
    except Exception:
        return False

MULT_PARAMS = [(n, k) for n, k in CANDIDATES if _has(k)]
NM = len(MULT_PARAMS)
OCV_NAMES = ["ocv_c0", "ocv_c1", "ocv_c2", "ocv_end_lo", "ocv_end_hi"]
NAMES = [n for n, _ in MULT_PARAMS] + ["R_contact"] + OCV_NAMES
LO = np.array([-1.0] * NM + [0.0] + [-1.0] * len(OCV_NAMES))
HI = np.array([1.0] * NM + [2.0] + [1.0] * len(OCV_NAMES))
STEP = np.array([0.1139] * NM + [0.4] + [0.25] * len(OCV_NAMES))


def _scaled(base, name):
    m = pybamm.InputParameter(name)
    if callable(base):
        return lambda *a, **k: base(*a, **k) * m
    return base * m


def positive_ocp_corrected(sto):
    """Step 3 positive OCP + dU(s) = c0 + c1(1-s) + c2(1-s)^2 + e_lo*exp(-s/w) + e_hi*exp(-(1-s)/w)  [mV]
    (s = SOC in the fitted window; all coefficients 0 -> exactly Step 3)."""
    s = (sto - y_0) / (y_100 - y_0)
    c0, c1, c2, e_lo, e_hi = (pybamm.InputParameter(n) for n in OCV_NAMES)
    return positive_electrode_ocp_safari(sto) + (
        c0 + c1 * (1 - s) + c2 * (1 - s) ** 2 + e_lo * pybamm.exp(-s / END_WIDTH) + e_hi * pybamm.exp(-(1 - s) / END_WIDTH)
    ) / 1000.0

def inputs_from_u(u):
    d = {n: float(10.0 ** ui) for (n, _), ui in zip(MULT_PARAMS, u[:NM])}
    d["R_contact"] = float(u[NM] * R_UNIT)
    for nm, ui, lim in zip(OCV_NAMES, u[NM + 1:], ALIGN_LIMITS_MV):
        d[nm] = float(ui * lim)
    return d

def physical_from_u(u):
    """Everything Step 5 needs, as ABSOLUTE values ready for parameter_values.update() (functions noted as such)."""
    out = {}
    for (n, k), ui in zip(MULT_PARAMS, u[:NM]):
        mult = float(10.0 ** ui)
        out[n] = {"parameter": k, "multiplier": mult, "is_function": bool(callable(parameter_values[k]))}
        if not callable(parameter_values[k]):
            out[n]["absolute_value"] = float(parameter_values[k]) * mult
    out["R_contact_ohm"] = float(u[NM] * R_UNIT)
    out["ocv_correction_mV"] = {nm: float(ui * lim) for nm, ui, lim in zip(OCV_NAMES, u[NM + 1:], ALIGN_LIMITS_MV)}
    return out

def eval_grid(t_data, i_data, n=N_SOLVE_PTS):
    m = len(t_data)
    idx = np.linspace(0, m - 1, min(n, m)).astype(int)
    di = np.abs(np.diff(i_data))
    thr = 0.02 * max(float(np.max(np.abs(i_data))), 1e-9)
    edge = np.where(di > thr)[0]
    idx = np.unique(np.r_[idx, edge, edge + 1, m - 1])
    return t_data[idx]

class SheetCase:
    def __init__(self, sheet_name):
        self.name = sheet_name
        self.df = load_sheet_data(sheet_name)
        self.t_full = self.df["t"].to_numpy(); self.v_full = self.df["V"].to_numpy(); self.i_full = self.df["I_raw"].to_numpy()
        self.soc0 = float(np.clip(soc_from_model_ocv(float(self.df["V"].iloc[0])), 0.0, 1.0))
        self.x_init = x_0 + self.soc0 * (x_100 - x_0); self.y_init = y_0 + self.soc0 * (y_100 - y_0)
        solve_df = decimate_rests(self.df, rest_dt_target=5.0)
        self.t_data = solve_df["t"].to_numpy(); self.i_data = solve_df["I_pybamm"].to_numpy()
        self.grid = eval_grid(self.t_data, self.i_data)
        n = len(self.t_full)
        self.idx = np.unique(np.r_[np.linspace(0, n - 1, min(N_EVAL_PTS, n)).astype(int), n - 1])
        self.w = 1.0 / np.sqrt(len(self.idx))
        self._sim = None

    def _make_pv(self):
        pv = parameter_values.copy()
        pv.update({
            "Initial concentration in negative electrode [mol.m-3]": self.x_init * pv["Maximum concentration in negative electrode [mol.m-3]"],
            "Initial concentration in positive electrode [mol.m-3]": self.y_init * pv["Maximum concentration in positive electrode [mol.m-3]"],
        })
        pv["Current function [A]"] = pybamm.Interpolant(self.t_data, self.i_data, pybamm.t, name=f"cur_{self.name}_{RUN_TAG}")
        for n, k in MULT_PARAMS:
            pv[k] = _scaled(pv[k], n)
        pv["Contact resistance [Ohm]"] = pybamm.InputParameter("R_contact")
        pv["Positive electrode OCP [V]"] = positive_ocp_corrected
        return pv

    def _sim_(self):
        if self._sim is None:
            self._sim = pybamm.Simulation(model_opt, parameter_values=self._make_pv(), solver=make_solver([VOLT], fit=True))
        return self._sim

    def resid(self, u):
        try:
            sol = self._sim_().solve(t_eval=self.grid, inputs=inputs_from_u(u))
            ts, vs = sol["Time [s]"].entries, sol[VOLT].entries
            tt = self.t_full[self.idx]
            r = np.interp(tt, ts, vs) - self.v_full[self.idx]
            late = tt > ts[-1]
            r[late] = np.clip(r[late], -TAIL_CAP_V, TAIL_CAP_V)
            return r * self.w
        except Exception as exc:
            if not getattr(self, "_warned", False):
                print(f"  [solve failed for {self.name}]: {str(exc)[:100]}", flush=True)
                self._warned = True
            return np.full(len(self.idx), 0.5) * self.w

def ocv_violation_V(u):
    """Size [V] of any decrease of the corrected full-cell OCV along SOC (0 if monotonic)."""
    c = np.asarray(u[NM + 1:], float) * ALIGN_LIMITS_MV
    sv = check_soc
    g = model_ocv_np(result.x, sv) + (
        c[0] + c[1] * (1 - sv) + c[2] * (1 - sv) ** 2 + c[3] * np.exp(-sv / END_WIDTH) + c[4] * np.exp(-(1 - sv) / END_WIDTH)
    ) / 1000.0
    return float(np.sum(np.maximum(0.0, -np.diff(g))))


def fit_ocv_start():
    """Numpy-only starting guess for the OCV correction, from the end-of-rest points of the TRAINING sheets
    (coulomb counting from each sheet's own backtracked initial SOC - same rule as Step 3)."""
    S, Vm = [], []
    for name in SHEETS:
        df = load_sheet_data(name)
        t, v, i = df["t"].to_numpy(), df["V"].to_numpy(), df["I_raw"].to_numpy()
        s0 = float(np.clip(soc_from_model_ocv(float(v[0])), 0.0, 1.0))
        sign = -1.0 if name.lower().startswith("discharge") else 1.0
        q = np.concatenate([[0.0], np.cumsum(0.5 * (i[1:] + i[:-1]) * np.diff(t))]) / 3600.0
        s = np.clip(s0 + sign * np.abs(q) / TARGET_CAPACITY_AH, 0.0, 1.0)
        rest = np.abs(i) < 1e-3
        edges = np.flatnonzero(np.diff(rest.astype(int)))
        starts = np.r_[0, edges + 1]; ends = np.r_[edges, len(t) - 1]
        for a, e in zip(starts, ends):
            if rest[a] and t[e] - t[a] >= ALIGN_MIN_REST_S:
                S.append(s[e]); Vm.append(v[e])
    S, Vm = np.array(S), np.array(Vm)
    if len(S) < 4:
        print("OCV-correction start skipped: fewer than 4 usable rests found.")
        return np.zeros(5)

    def dU(c, sv):
        return (c[0] + c[1] * (1 - sv) + c[2] * (1 - sv) ** 2) / 1000.0

    def res(c):
        return np.r_[model_ocv_np(result.x, S) + dU(c, S) - Vm, ALIGN_RIDGE * np.asarray(c) / 1000.0]
    c3 = least_squares(res, np.zeros(3), bounds=(-ALIGN_LIMITS_MV[:3], ALIGN_LIMITS_MV[:3])).x
    c = np.r_[c3, 0.0, 0.0]
    if not np.all(np.diff(model_ocv_np(result.x, check_soc) + dU(c[:3], check_soc)) > 0):
        c = np.zeros(5)
    before = float(np.sqrt(np.mean((model_ocv_np(result.x, S) - Vm) ** 2)))
    after = float(np.sqrt(np.mean((model_ocv_np(result.x, S) + dU(c[:3], S) - Vm) ** 2)))
    print(f"OCV-correction start from {len(S)} rests (training sheets): c = {np.round(c, 1)} mV | "
          f"rest RMSE {before * 1000:.1f} -> {after * 1000:.1f} mV", flush=True)
    return c

def u_ocv_start():
    u = np.zeros(len(NAMES))
    if os.path.exists(ALIGN_JSON):
        with open(ALIGN_JSON) as fh:
            c = np.array(json.load(fh)["c_mV"], dtype=float)
    else:
        c = fit_ocv_start()
        os.makedirs(STATE_DIR, exist_ok=True)
        with open(ALIGN_JSON, "w") as fh:
            json.dump({"c_mV": [float(x) for x in c]}, fh, indent=2)
    u[NM + 1:] = c / ALIGN_LIMITS_MV
    return u

def state_signature():
    sig = {"NEG_DIFFUSIVITY": NEG_DIFFUSIVITY, "TARGET_CAPACITY_AH": TARGET_CAPACITY_AH,
           "window": [round(float(v), 6) for v in (x_0, x_100, y_0, y_100)],
           "names": NAMES, "lo": LO.tolist(), "hi": HI.tolist(), "sheets": SHEETS,
           "N_EVAL_PTS": N_EVAL_PTS, "N_SOLVE_PTS": N_SOLVE_PTS, "FIT_ATOL": FIT_ATOL, "FIT_RTOL": FIT_RTOL,
           "raw_xlsx_bytes": os.path.getsize(RAW_XLSX) if os.path.exists(RAW_XLSX) else -1,
           "ocv_xlsx_bytes": os.path.getsize(OCV_XLSX) if os.path.exists(OCV_XLSX) else -1}
    return hashlib.md5(json.dumps(sig, sort_keys=True).encode()).hexdigest()

def check_state_meta():
    os.makedirs(EVAL_DIR, exist_ok=True)
    sig = state_signature()
    if os.path.exists(META_JSON):
        with open(META_JSON) as fh:
            old = json.load(fh).get("signature")
        if old != sig:
            raise SystemExit("Saved progress in outputs/step4_state used DIFFERENT settings/data.\n"
                             "Run:  python step4_optimize.py --reset   and start again.")
    else:
        with open(META_JSON, "w") as fh:
            json.dump({"signature": sig, "created": time.ctime()}, fh)

class EvalCache:
    def __init__(self):
        self.index = {}
        if os.path.exists(EVAL_INDEX):
            with open(EVAL_INDEX) as fh:
                for line in fh:
                    try:
                        d = json.loads(line); self.index[d["key"]] = d
                    except Exception:
                        pass

    @staticmethod
    def key(u):
        return hashlib.md5(np.round(np.asarray(u, float), 9).tobytes()).hexdigest()

    def get(self, u):
        k = self.key(u)
        if k in self.index:
            p = os.path.join(EVAL_DIR, k + ".npy")
            if os.path.exists(p):
                try:
                    return np.load(p)
                except Exception:
                    return None
        return None

    def put(self, u, r):
        k = self.key(u)
        tmp = os.path.join(EVAL_DIR, k + ".tmp.npy")
        np.save(tmp, r); os.replace(tmp, os.path.join(EVAL_DIR, k + ".npy"))
        rec = {"key": k, "u": [float(x) for x in u], "cost": float(0.5 * np.sum(r ** 2))}
        with open(EVAL_INDEX, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        self.index[k] = rec

    def best(self):
        return min(self.index.values(), key=lambda d: d["cost"]) if self.index else None

def optimize_pipeline(ev, set_stage):
    """Non-gradient sensitivity scan -> non-gradient global sample -> gradient-based local
    refinement (Broyden-updated Jacobian, see the module docstring above)."""
    n = len(NAMES); u0 = np.zeros(n)
    sens = {}; act = np.ones(n, dtype=bool); cost0 = float("nan")
    try:
        set_stage("start"); f0 = ev(u0); cost0 = 0.5 * np.sum(f0 ** 2)

        set_stage("A sensitivity (non-gradient)")
        cols = []
        for i, nm in enumerate(NAMES):
            d = np.zeros(n); d[i] = STEP[i]
            fp = ev(u0 + d)
            cols.append((fp - f0) / STEP[i])
            sens[nm] = float(np.linalg.norm(fp - f0))
        J_u0 = np.column_stack(cols)
        smax = max(max(sens.values()), 1e-12)
        act = np.array([(sens[nm] >= FREEZE_FRACTION * smax) or nm == "R_contact" for nm in NAMES])

        def full(z):
            u = np.zeros(n); u[act] = z; return u
        lo, hi = LO[act], HI[act]; nd = int(act.sum())

        set_stage("B global sample (non-gradient, Latin Hypercube) + OCV start")
        rng = np.random.default_rng(0)
        lhs = (np.array([rng.permutation(N_LHS) for _ in range(nd)]).T + rng.random((N_LHS, nd))) / N_LHS
        cand = lo + lhs * (hi - lo)
        costs = [0.5 * np.sum(ev(full(c)) ** 2) for c in cand]
        cand_list = [np.zeros(nd)] + list(cand); cost_list = [cost0] + costs
        z_a = u_ocv_start()[act]
        cand_list.append(z_a); cost_list.append(0.5 * np.sum(ev(full(z_a)) ** 2))
        z_start = cand_list[int(np.argmin(cost_list))]

        set_stage("C local refinement (gradient-based, Broyden)")
        state = {"z": z_start.copy(), "f": ev(full(z_start)), "J": J_u0[:, act].copy(), "acc": 0}

        def fd_jacobian(z):
            fz = ev(full(z)); J = np.empty((len(fz), nd))
            for j in range(nd):
                h = 0.05 if z[j] + 0.05 <= hi[j] else -0.05
                zz = z.copy(); zz[j] += h
                J[:, j] = (ev(full(zz)) - fz) / h
            return J

        def jac_fn(z):
            if np.array_equal(z, state["z"]):
                return state["J"]
            f_new = ev(full(z)); dz = z - state["z"]; df = f_new - state["f"]
            state["J"] = state["J"] + np.outer(df - state["J"] @ dz, dz) / max(float(dz @ dz), 1e-12)
            state["z"], state["f"], state["acc"] = z.copy(), f_new, state["acc"] + 1
            if state["acc"] % JAC_REFRESH_EVERY == 0:
                state["J"] = fd_jacobian(z)
            return state["J"]

        def res_with_penalty(z):
            return np.r_[ev(full(z)), ocv_violation_V(full(z)) * 50.0]

        def jac_with_penalty(z):
            return np.vstack([jac_fn(z), np.zeros((1, nd))])

        least_squares(res_with_penalty, z_start, jac=jac_with_penalty, bounds=(lo, hi), method="trf",
                      ftol=1e-4, xtol=1e-3, max_nfev=MAX_TRF_NFEV)
    except CapReached:
        print("Evaluation cap reached - using the best point found so far.", flush=True)
    return {"sens": sens, "act": act.tolist(), "cost0": float(cost0)}

def worker(args):
    lower_priority()
    check_state_meta()
    cache = EvalCache()
    t_start = time.time()
    holder = {"cases": None, "stage": "start", "n_new": 0}

    def get_cases():
        if holder["cases"] is None:
            print("  (worker) loading data and building the DFN models ...", flush=True)
            holder["cases"] = [SheetCase(s) for s in SHEETS]
        return holder["cases"]

    def ev(u):
        u = np.asarray(u, float)
        r = cache.get(u)
        if r is not None:
            return r
        if len(cache.index) >= args.max_evals:
            raise CapReached()
        if (time.time() - t_start > args.chunk_seconds or holder["n_new"] >= MAX_EVALS_PER_CHUNK
                or (psutil is not None and rss_mb() > args.mem_limit_mb)):
            raise BudgetExhausted()
        cases = get_cases()
        t0 = time.time()
        r = np.concatenate([c.resid(u) for c in cases])
        secs = time.time() - t0
        cache.put(u, r)
        holder["n_new"] += 1
        b = cache.best()
        print(f"  [eval {len(cache.index):3d}/{args.max_evals} | {holder['stage']:38s}] cost {0.5 * np.sum(r ** 2):.4e} "
              f"(best {b['cost']:.4e}) | {secs:5.1f} s | RAM {rss_mb():.0f} MB", flush=True)
        return r

    def set_stage(s):
        holder["stage"] = s

    try:
        info = optimize_pipeline(ev, set_stage)
    except BudgetExhausted:
        print(f"  chunk finished ({len(cache.index)} evaluations saved) - starting a fresh worker.", flush=True)
        sys.exit(10)
    b = cache.best()
    u_best = np.array(b["u"])
    act = np.array(info["act"])
    at_bound = [nm for nm, ui, l, h, a in zip(NAMES, u_best, LO, HI, act)
                if a and (abs(ui - l) < 0.02 * (h - l) or abs(ui - h) < 0.02 * (h - l))]
    res = {"u": u_best.tolist(), "names": NAMES, "frozen": [nm for nm, a in zip(NAMES, act) if not a],
           "at_bounds": at_bound, "sensitivity": info["sens"], "n_evaluations": len(cache.index),
           "cost_start": info["cost0"], "cost_best": b["cost"]}
    with open(RESULT_JSON, "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"  optimization finished: {len(cache.index)} evaluations, cost {info['cost0']:.4e} -> {b['cost']:.4e}", flush=True)
    sys.exit(0)

def write_final_parameters():
    with open(RESULT_JSON) as fh:
        res = json.load(fh)
    u = np.array(res["u"])
    phys = physical_from_u(u)
    out = {
        "description": "Optimized parameters for the A123 ANR26650M1B DFN model. Apply with "
                       "parameter_values.update() in step5_validate.py; function-type parameters "
                       "(exchange-current densities, electrolyte conductivity, positive electrode OCP) "
                       "are updated with a small wrapper function, since PyBaMM represents them as "
                       "functions rather than single numbers - this is the one necessary exception "
                       "to a plain .update() call.",
        "fixed_from_step3": {
            "target_capacity_Ah": TARGET_CAPACITY_AH, "neg_diffusivity_baseline_m2_s": NEG_DIFFUSIVITY,
            "stoich_window": {"x_0": x_0, "x_100": x_100, "y_0": y_0, "y_100": y_100},
            "voltage_cutoffs_V": [2.0, 3.8], "eps_n": eps_n_new, "eps_p": eps_p_new,
        },
        "optimized": phys,
        "frozen_not_identified": res["frozen"],
        "at_bounds_treat_with_caution": res["at_bounds"],
        "sensitivity": res["sensitivity"],
        "n_evaluations": res["n_evaluations"], "cost_start": res["cost_start"], "cost_best": res["cost_best"],
        "ocv_monotonic": ocv_violation_V(u) == 0.0,
        "training_sheets": SHEETS,
    }
    with open(FINAL_JSON, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    return out

def run_child(flag, extra_args=(), timeout=None):
    cmd = [sys.executable, os.path.abspath(__file__), flag] + list(extra_args)
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    try:
        return subprocess.run(cmd, timeout=timeout, **kw).returncode
    except subprocess.TimeoutExpired:
        print("  (a worker exceeded its time limit and was stopped; progress is saved)", flush=True)
        return 124

def orchestrate(args):
    check_state_meta()
    fwd = ["--chunk-seconds", str(args.chunk_seconds), "--max-evals", str(args.max_evals), "--mem-limit-mb", str(args.mem_limit_mb)]
    crashes = 0
    while not os.path.exists(RESULT_JSON):
        rc = run_child("--worker", fwd, timeout=args.chunk_seconds + 1800)
        if rc == 10:
            crashes = 0; gc.collect(); continue
        if rc == 0:
            break
        crashes += 1
        print(f"  worker stopped unexpectedly (code {rc}); {crashes}/3. Progress is saved.", flush=True)
        if crashes >= 3:
            raise SystemExit("Three workers in a row failed. Progress is saved in outputs/step4_state - "
                             "close other programs / lower --mem-limit-mb and run the same command again to resume.")
        time.sleep(3)
    out = write_final_parameters()
    print("\n" + "=" * 78 + f"\nOptimization done: {out['n_evaluations']} simulations, cost {out['cost_start']:.4e} -> {out['cost_best']:.4e}", flush=True)
    if out["at_bounds_treat_with_caution"]:
        print("WARNING - at their bounds (not identified from this data):", out["at_bounds_treat_with_caution"])
    if not out["ocv_monotonic"]:
        print("WARNING - corrected OCV is not monotonic; re-run with --reset.")
    print(f"Saved: {os.path.abspath(FINAL_JSON)}\nRun step5_validate.py next.", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--chunk-seconds", type=float, default=DEFAULT_CHUNK_SECONDS)
    ap.add_argument("--max-evals", type=int, default=DEFAULT_MAX_EVALS)
    ap.add_argument("--mem-limit-mb", type=float, default=DEFAULT_MEM_LIMIT_MB)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    if args.reset:
        shutil.rmtree(STATE_DIR, ignore_errors=True)
        print("Saved optimization progress deleted.")
        return
    if args.worker:
        worker(args)
    else:
        print(f"Optimization variables ({len(NAMES)}): {NAMES}")
        orchestrate(args)

if __name__ == "__main__":
    builtins.print = _real_print
    main()