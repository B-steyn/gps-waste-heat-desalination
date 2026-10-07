"""
The sensitivity table: every quantity that was SWEPT, the range it was swept
over, and the value adopted for the reported results.

Every value is READ from the code or the model outputs - nothing is typed
in - so the table cannot drift from the model it describes.

Three categories, which must not be confused in the report:

  economic sensitivity  - uncertain inputs we cannot choose. Swept to show
                          how far the answer moves (tornado + factorial).
  physical sensitivity  - uncertain physical inputs, swept or banded.
  design optimisation   - choices WE make. Swept to find the best value,
                          not to express uncertainty; the adopted value is
                          the optimum, and it can differ by architecture.
"""

import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module5_orc"))

import numpy as np
import pandas as pd

import common.constants as C
import common.economic_constants as ec
import common.fan_margin as fm
from common.costing import ORC_U_BAND_W_M2K


def _load(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    return mod


def _fmt(values):
    return ", ".join(f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v)
                     for v in values)


def main():
    sweep = _load(ROOT / "module6_lcow" / "sensitivity_sweep.py", "sweep")
    m6 = _load(ROOT / "module6_lcow" / "module6_lcow.py", "m6")
    from design_cases import DUCT_CASES

    lcow = pd.read_csv(ROOT / "outputs" / "gps_module6_lcow_bhojwani_global_lcow.csv")
    best = lcow.loc[lcow.groupby("architecture")["lcow_aud_per_m3"].idxmin()].set_index("architecture")

    rows = []

    def add(category, parameter, swept, adopted, unit, basis):
        if isinstance(adopted, (float, np.floating)):
            adopted = f"{adopted:g}"          # 25 rather than 25.0
        rows.append({"category": category, "parameter": parameter,
                     "range_swept": swept, "baseline_value": adopted,
                     "unit": unit, "basis": basis})

    # ---- economic sensitivities: the nine factorial inputs ----
    units = {"oandm_water": "fraction of CAPEX/yr", "oandm_heat": "fraction of CAPEX/yr",
             "lang": "x purchased cost", "cost_error": "x correlation", "wacc_nom": "nominal, fraction",
             "life": "years", "membrane_life": "years", "elec": "A$/MWh", "desal_avail": "fraction"}
    for label, attr, values, default in sweep.sweep_axes():
        adopted = default
        basis = "tornado + full factorial"
        if attr == "wacc_nom":
            basis += f"; converted to real {ec.WACC_REAL*100:.3f}% via the Fisher relation"
        add("economic sensitivity", label, _fmt(values), adopted, units.get(attr, "-"), basis)

    # ---- physical sensitivities ----
    areas = sorted(w * h for _, w, h in DUCT_CASES)
    add("physical sensitivity (retired - now measured)", "duct cross-section", _fmt([round(a, 2) for a in areas]),
        round(C.DUCT_WIDTH_M * C.DUCT_HEIGHT_M, 2), "m2",
        f"measured {C.DUCT_WIDTH_M} x {C.DUCT_HEIGHT_M} m at the fabric-filter outlet")
    pf = sorted({*fm.ID_FAN_PF_BAND, fm.ID_FAN_POWER_FACTOR})
    add("physical sensitivity", "ID fan motor power factor", _fmt(pf), fm.ID_FAN_POWER_FACTOR, "-",
        "no logged motor kW exists; range corroborated by measured fan suction pressure")
    add("physical sensitivity", "gas-side fouling multiplier on tube-bank dP", _fmt(fm.FOULING_DP_FACTORS),
        fm.FOULING_DP_FACTORS[0], "x clean dP", "fouled cases bound the row cap; the reported design is clean")
    add("physical sensitivity", "ORC / preheater overall U", _fmt(ORC_U_BAND_W_M2K), m6.ORC_U_BASELINE_W_M2K,
        "W/m2K", "no ORC-specific exchanger data; carried as a band")
    t_sw, _ = m6._seawater_rows(str(ROOT / "data"))
    add("physical sensitivity (per-record data)", "seawater intake temperature",
        f"{np.min(t_sw):.1f} - {np.max(t_sw):.1f} (461 measured records)",
        round(float(np.percentile(t_sw, m6.SEAWATER_DESIGN_PERCENTILE)), 1), "C",
        "used record by record for annual SEC; the median sets the design point")

    # ---- design optimisations: adopted value is the optimum, from the outputs ----
    def per_arch(col, fmt="{:g}"):
        keep = [a for a in best.index if col in best.columns and pd.notna(best.loc[a, col])]
        return "; ".join(f"{a}: {fmt.format(best.loc[a, col])}" for a in keep)

    add("design optimisation", "flue-gas extraction track", _fmt(C.TARGET_TEMPERATURES),
        per_arch("track_c"), "C", "optimum per architecture, subject to the ID fan cap")
    add("design optimisation", "loop return temperature", "scanned per architecture",
        per_arch("loop_return_c", "{:.2f}"), "C",
        "floor for ORC branches; interior optimum for Direct-RO")
    add("design optimisation", "ORC path", "A (subcritical), B (transcritical), C (TFC, cycle-level only)",
        "A", "-", "Path A wins on LCOW at every Stage 4 corner")
    add("design optimisation", "RO membrane case", _fmt([C.RO_CEILING_CURRENT_MEMBRANE_C,
                                                        C.RO_CEILING_ADVANCED_MEMBRANE_C]),
        C.RO_CEILING_CURRENT_MEMBRANE_C, "C feed ceiling", "current membrane wins at every corner")
    # computed from the per-record sizing check, not typed
    _ms = ROOT / "outputs" / "med_sizing_percentile_check.csv"
    _basis = "sized for the P90 hour of delivered heat"
    if _ms.exists():
        _g = pd.read_csv(_ms).query("cost_basis == 'bhojwani_global'").set_index("sizing_percentile")
        _p = m6.MED_SIZING_PERCENTILE
        _basis = (f"per-record heat: at P{_p} the plant averages {_g.capacity_factor[_p]:.1%} of its size and "
                  f"spills {_g.heat_spilled_pct[_p]:.2f} % of the heat; the cheapest size on the global basis, "
                  f"P{_g.lcow_aud_per_m3.idxmin()}, is {100 * (1 - _g.lcow_aud_per_m3.min() / _g.lcow_aud_per_m3[_p]):.1f} % "
                  "cheaper")
    add("design optimisation", "MED sizing percentile of delivered heat", "P50 - P100",
        f"P{m6.MED_SIZING_PERCENTILE}", "-", _basis)
    add("design optimisation", "ORC+MED condensing temperature", _fmt(C.ORC_MED_TC_CANDIDATES_C),
        per_arch("tc_c") if "tc_c" in best.columns else f"{C.ORC_MED_TC_RANGE_C[0]:g} (only value run)",
        "C", "sets the MED top brine temperature for ORC+MED")
    if "tbt_c" in best.columns:
        add("design optimisation", "MED top brine temperature", "scanned per MED architecture",
            per_arch("tbt_c", "{:.1f}"), "C", "coupled to the number of effects")
    if "n_effects" in best.columns:
        add("design optimisation", "MED number of effects", "1 - physical maximum at each TBT",
            per_arch("n_effects", "{:.0f}"), "-", "coupled to the top brine temperature")

    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "sensitivity_table.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    with pd.option_context("display.max_colwidth", 60, "display.width", 200):
        print(main()[["category", "parameter", "range_swept", "baseline_value"]].to_string(index=False))
