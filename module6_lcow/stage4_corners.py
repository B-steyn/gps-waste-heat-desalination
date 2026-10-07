"""
Stage 4 of the sensitivity sweep: does the OPTIMAL DESIGN itself move at
the corners of the sensitivity space?

Stage 3 holds the design fixed and varies the economics, which bounds the
ANSWER. This stage re-optimises the ORC+RO design at each corner, which
tests whether the DESIGN is stable.

A moving optimum here is a FINDING, not a problem: "the recommended
configuration depends on an unverified input" is a legitimate and
important conclusion, and one a reviewer will look for.

What is re-optimised at each corner: track, ORC path, membrane case. The
loop return is set by bisection to its minimum feasible value, valid
because LCOW rises monotonically with loop return
(common.loop_optimisation.find_minimum_feasible_loop_return).

What defines a corner: the parameters that could plausibly change the
DESIGN rather than just its cost. Duct dimensions change the
thermodynamics through the fan and the heat-transfer coefficient. WACC
extremes reweight capital against operating cost, which is exactly the
trade the loop return and track make. 5 ducts x 3 rates = 15 corners.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "module5_orc"))

import importlib.util
import itertools
from functools import partial

import pandas as pd

from common.gas_properties import dynamic_flue_gas_cp
from common.hx_loop_physics import duct_geometry, gas_side_heat_transfer
from common.loop_optimisation import find_minimum_feasible_loop_return
from common.fgche_sizing import size_fgche
from common.constants import (
    N_GAS_TRAINS, N_TRAINS_FITTED,
    DUCT_HEIGHT_M, PITCH_RATIO, LOSS_BUFFER_DELTA_T, HX_SHELL_EFFICIENCY,
    ORC_EVAPORATOR_PINCH_C, TARGET_TEMPERATURES,
    ORC_RO_TC_CURRENT_MEMBRANE_C, ORC_RO_TC_ADVANCED_MEMBRANE_C,
)
import common.economic_constants as ec
from path_a_subcritical import solve_path_a
from path_b_transcritical import solve_path_b, ORC_TRANS_FLUID
from design_cases import load_design_row, DUCT_CASES
from common.costing import area_from_pinch_profile_m2
from common.fan_margin import row_caps_for_ducts

_sp = importlib.util.spec_from_file_location(
    "m6", Path(__file__).resolve().parent / "module6_lcow.py")
m6 = importlib.util.module_from_spec(_sp)
_sp.loader.exec_module(m6)

MEMBRANE_CASES = {"current": ORC_RO_TC_CURRENT_MEMBRANE_C,
                  "advanced": ORC_RO_TC_ADVANCED_MEMBRANE_C}
_DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")

# corners: the parameters that could move the DESIGN, not just its cost
CORNERS = {
    "duct": [c[0] for c in DUCT_CASES],
    "anchor": [ec.DESAL_COST_ANCHOR],
    "wacc_nominal": [0.05, 0.08, 0.13],
}


_ROW_CAPS = None


def best_design_at_corner(design_row, duct_label, anchor, wacc_nominal, row_caps=None):
    """Re-optimise track / path / membrane at one corner; return the winner."""
    dw, dh = next((w, h) for (l, w, h) in DUCT_CASES if l == duct_label)
    n_tubes, a_ref = duct_geometry(dw, PITCH_RATIO)
    gas_transport = gas_side_heat_transfer(
        design_row["t_in_c"], design_row["m_dot_fg"], design_row["p_gauge"],
        design_row["dust"], a_ref * (dh / DUCT_HEIGHT_M))

    r_real = (1 + wacc_nominal) / (1 + ec.INFLATION) - 1
    candidates = []

    for t_target, membrane, path in itertools.product(
            TARGET_TEMPERATURES, MEMBRANE_CASES, ("A", "B")):
        if design_row["t_in_c"] <= t_target:
            continue
        tc = MEMBRANE_CASES[membrane]
        ceiling = 38.0 if membrane == "current" else 50.0

        cp_d = dynamic_flue_gas_cp((design_row["t_in_c"] + t_target) / 2.0,
                                    design_row["mole_fractions"])
        q_target = (design_row["m_dot_fg"] * (N_TRAINS_FITTED / N_GAS_TRAINS) * cp_d * (design_row["t_in_c"] - t_target))
        q_in = q_target * HX_SHELL_EFFICIENCY
        t_hot = t_target - LOSS_BUFFER_DELTA_T
        c_gas = (design_row["m_dot_fg"] / N_GAS_TRAINS) * cp_d

        if path == "A":
            solver = lambda hh, hr, md, _q=q_in, _tc=tc, _th=t_hot: solve_path_a(
                t_evap_unconstrained_c=_th - ORC_EVAPORATOR_PINCH_C, t_cond_c=_tc,
                q_in_kw=_q, h_loop_hot=hh, h_loop_return=hr, m_dot_loop=md,
                fluid=ORC_TRANS_FLUID)
        else:
            solver = lambda hh, hr, md, _q=q_in, _tc=tc, _th=t_hot: solve_path_b(
                t3_search_low_c=_tc + 3.0, t3_search_high_c=_th - 2.0, t_cond_c=_tc,
                q_in_kw=_q, h_loop_hot=hh, h_loop_return=hr, m_dot_loop=md, n_scan=30)

        # Fan-cap pre-check BEFORE the expensive ORC solve. Tube
        # rows only grow as the loop return rises - a warmer return narrows the
        # cold-end approach and needs more area - so if even the lowest allowed
        # return needs more rows than the fan can carry, no return can pass.
        # Results are identical; it just skips the full ORC bisection on
        # combinations that could never be feasible.
        cap = (row_caps or {}).get(duct_label)
        if cap is not None:
            floor = size_fgche(tc + 2.0, t_hot, q_target, design_row["t_in_c"], c_gas,
                               gas_transport, n_tubes, dh)
            if floor is not None and floor["rows_316"] > cap:
                continue

        e = find_minimum_feasible_loop_return(
            solver, t_hot, q_target, design_row["t_in_c"], c_gas, gas_transport,
            n_tubes, dh, lr_lower_bound_c=tc + 2.0, t_gas_out_c=t_target)
        if e is None:
            continue
        # ID fan constraint at the chosen loop return - a design needing a
        # deeper bank than the fan can carry is not a candidate at all
        if cap is not None and e["rows"] > cap:
            continue

        # the same per-record seawater treatment as Module 6
        perf = m6.ro_annual_performance(e["q_cond_kw"], ceiling, _DATA_DIR)
        if perf is None:
            continue
        sec_pre = perf["sec_preheated_kwh_m3"]
        sec_amb = perf["sec_ambient_kwh_m3"]
        water = perf["water_m3_per_day"]

        # the ORC exchangers are costed exactly as in Module 6
        ua = area_from_pinch_profile_m2(e["pinch"]["profile"], e["pinch"]["q_total_w"], 1.0)
        r = m6.assemble_architecture(
            f"orc_ro_{membrane} (path {path})", "SWRO", e["q_cond_kw"], water,
            e["area_total_m2"], e["fan_kw"], e["loop_pump_kw"], e["w_net_kw"],
            sec_pre, sec_amb, anchor=anchor, real_rate=r_real,
            orc_evap_ua_w_per_k=ua, orc_t_cond_c=tc,
            orc_sink_in_c=perf["design_seawater_c"], orc_sink_out_c=ceiling)
        if r["lcow_aud_per_m3"] is None:
            continue
        candidates.append({
            "duct": duct_label, "anchor": anchor, "wacc_nominal": wacc_nominal,
            "track_c": t_target, "membrane": membrane, "path": path,
            "loop_return_c": e["loop_return_c"], "water_m3_per_day": water,
            "orc_gross_kw": e["w_net_kw"], "fan_kw": e["fan_kw"],
            "lcow_aud_per_m3": r["lcow_aud_per_m3"],
        })

    if not candidates:
        return None
    return min(candidates, key=lambda c: c["lcow_aud_per_m3"])


def _corner(args):
    """Worker: re-optimise one corner in its own process."""
    design_row, duct, anchor, wacc, caps = args
    return duct, anchor, wacc, best_design_at_corner(design_row, duct, anchor, wacc, row_caps=caps)


def main():
    root = Path(__file__).resolve().parent.parent
    d = load_design_row(str(root / "data"))
    corners = list(itertools.product(CORNERS["duct"], CORNERS["anchor"],
                                      CORNERS["wacc_nominal"]))
    print(f"Stage 4: re-optimising design at {len(corners)} corners\n", flush=True)

    caps = row_caps_for_ducts(str(root / "data"), DUCT_CASES)
    print("ID fan row caps: " + ", ".join(f"{k}={v}" for k, v in caps.items()) + "\n", flush=True)

    # Corners are independent, so they run in parallel (about 2 minutes a
    # corner if run serially).
    import multiprocessing as mp
    with mp.Pool(min(len(corners), mp.cpu_count())) as pool:
        results = pool.map(_corner, [(d, duct, anchor, wacc, caps) for duct, anchor, wacc in corners])

    rows = []
    for duct, anchor, wacc, best in results:
        if best is None:
            print(f"  {duct:28s} {anchor:16s} wacc={wacc:.2f}  no feasible design", flush=True)
            continue
        rows.append(best)
        print(f"  {duct:28s} {anchor:16s} wacc={wacc:.2f} -> "
              f"track {best['track_c']}C, {best['membrane']}, path {best['path']}, "
              f"A${best['lcow_aud_per_m3']:.3f}/m3", flush=True)

    df = pd.DataFrame(rows)
    out = root / "outputs" / "sweep_stage4_corners.csv"
    df.to_csv(out, index=False)

    print()
    print("=" * 72)
    print("DOES THE OPTIMAL DESIGN MOVE?")
    print("=" * 72)
    for col in ("track_c", "membrane", "path"):
        vals = df[col].value_counts()
        stable = len(vals) == 1
        print(f"  {col:12s}: {'STABLE - ' if stable else 'MOVES - '}"
              f"{dict(vals)}")
    print()
    print(f"  loop return: {df.loop_return_c.min():.2f}-{df.loop_return_c.max():.2f}C")
    print(f"  LCOW across corners: A${df.lcow_aud_per_m3.min():.3f}-{df.lcow_aud_per_m3.max():.3f}/m3")
    print(f"\nwrote {out}")
    return df


if __name__ == "__main__":
    main()
