"""
Tests of the pinch check (common/pinch_check.py), which evaluates both
streams on an even grid AND exactly at their bubble and dew points.

These test whether the check picks things up, not whether a design passes.
Each one feeds the check a known case and confirms it gives the right
answer:

  T1  an endpoint-passing design                         -> must FAIL: the streams cross
  T2  designs optimised with an even-grid check alone    -> must FAIL: below 5 K
  T3  the check against a brute-force 1,000-point check     -> the same minimum, within 0.02 K
  T4  the transcritical (path B) designs                     -> unchanged, still PASS
  T5  evaporator area: an even grid gives exactly the old integration, and the
      new grid lands within 1 % of a 1,000-slice integration

The even-grid designs are frozen in validation/fixtures/, so these tests
stay repeatable whatever the optimiser now produces.

Run:  python validation/pinch_check_tests.py   (exit code 1 if any test fails)
"""

import math
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "module5_orc"))

from common.constants import (
    N_GAS_TRAINS, N_TRAINS_FITTED, LOSS_BUFFER_DELTA_T, HX_SHELL_EFFICIENCY, PITCH_RATIO,
    DUCT_WIDTH_M, DUCT_HEIGHT_M, P_WATER_LOOP_PA, ORC_EVAPORATOR_PINCH_C,
    ORC_TURBINE_ISENTROPIC_EFF, ORC_PUMP_ISENTROPIC_EFF,
)
from common.costing import area_from_pinch_profile_m2
from common.fgche_sizing import size_fgche
from common.gas_properties import dynamic_flue_gas_cp
from common.hx_loop_physics import duct_geometry, gas_side_heat_transfer
from common.loop_optimisation import evaluate_loop_return
from common.orc_cycle import solve_orc_cycle
from common.pinch_check import discretized_pinch_check
from path_b_transcritical import ORC_TRANS_FLUID as FLUID
from design_cases import load_design_row
from loop_optimiser import _solver_b

OPT = pd.read_csv(HERE / "fixtures" / "pinch_pre_fix_optimiser_designs.csv")
OPT = OPT[OPT.duct.str.contains("MEASURED")]
M6 = pd.read_csv(HERE / "fixtures" / "pinch_pre_fix_module6.csv")

DESIGN = load_design_row(str(ROOT / "data"))
N_TUBES, A_FLOW = duct_geometry(DUCT_WIDTH_M, PITCH_RATIO)
GT = gas_side_heat_transfer(DESIGN["t_in_c"], DESIGN["m_dot_fg"], DESIGN["p_gauge"], DESIGN["dust"], A_FLOW)
RESULTS = []


def record(test, case, expected, got, ok):
    RESULTS.append((test, case, expected, got, "PASS" if ok else "** FAIL **"))


def context(track, lr):
    """module5_orc.loop_optimiser.run_case's own set-up, measured duct."""
    cp_d = dynamic_flue_gas_cp((DESIGN["t_in_c"] + track) / 2.0, DESIGN["mole_fractions"])
    q_target = DESIGN["m_dot_fg"] * (N_TRAINS_FITTED / N_GAS_TRAINS) * cp_d * (DESIGN["t_in_c"] - track)
    c = {"q_target": q_target, "q_in": q_target * HX_SHELL_EFFICIENCY,
         "c_gas": (DESIGN["m_dot_fg"] / N_GAS_TRAINS) * cp_d, "t_hot": track - LOSS_BUFFER_DELTA_T}
    c["sized"] = size_fgche(lr, c["t_hot"], q_target, DESIGN["t_in_c"], c["c_gas"], GT, N_TUBES,
                            DUCT_HEIGHT_M, p_loop_pa=P_WATER_LOOP_PA)
    return c


def check(cycle, sized, n=20, phase=True):
    sp = cycle["state_points"]
    return discretized_pinch_check(
        "Water", P_WATER_LOOP_PA, sized["h_loop_hot"], sized["h_loop_return"], sized["m_dot_loop_kg_s"],
        FLUID, cycle.get("p_evap_pa", cycle.get("p_high_pa")), sp[2]["h"], sp[3]["h"],
        cycle["m_dot_wf_kg_s"], n_segments=n, min_pinch_k=ORC_EVAPORATOR_PINCH_C,
        include_phase_change_points=phase)


def cycle_a(t_evap, tc, q_in):
    return solve_orc_cycle(t_evap_c=t_evap, t_cond_c=tc, q_in_kw=q_in, fluid=FLUID,
                           turbine_eff=ORC_TURBINE_ISENTROPIC_EFF, pump_eff=ORC_PUMP_ISENTROPIC_EFF)


def reported_designs():
    """The nine frozen ORC designs, as optimised with the even-grid check alone."""
    out = []
    for _, r in M6[M6.architecture.str.startswith("orc_")].iterrows():
        arch, path = r.architecture.split(" (path ")
        path = path.rstrip(")")
        key = arch
        if arch == "orc_med":
            key = "orc_med" if float(r.tc_c) == 65.0 else f"orc_med_tc{float(r.tc_c):.0f}"
        o = OPT[(OPT.architecture == key) & (OPT.track_c == r.track_c) & (OPT.path == path)].iloc[0]
        out.append((f"{r.architecture} {r.track_c:.0f} C", path, o))
    return out


def main():
    designs = reported_designs()

    # T1 - an endpoint-passing crossing at the reported ORC+RO design point
    o = next(o for name, p, o in designs if name.startswith("orc_ro_current (path A) 85"))
    c = context(85.0, float(o.loop_return_c))
    cyc = cycle_a(c["t_hot"] - ORC_EVAPORATOR_PINCH_C, float(o.tc_c), c["q_in"])
    new, old = check(cyc, c["sized"]), check(cyc, c["sized"], phase=False)
    record("T1", "endpoint-passing design, ORC+RO 85 C", "fail (cross)",
           f"{new['min_pinch_k']:+.2f} K (even-grid check {old['min_pinch_k']:+.2f} K)",
           (not new["valid"]) and new["min_pinch_k"] < 0.0)

    for name, path, o in designs:
        c = context(float(o.track_c), float(o.loop_return_c))
        if path == "A":
            cyc = cycle_a(float(o.operating_point_c), float(o.tc_c), c["q_in"])
            old20 = check(cyc, c["sized"], phase=False)
            # the rebuilt design must be exactly the one the optimiser reported
            same = (math.isclose(cyc["w_net_kw"], float(o.w_net_kw), rel_tol=1e-9)
                    and abs(old20["min_pinch_k"] - float(o.min_pinch_k)) < 1e-6)
            new = check(cyc, c["sized"])
            record("T2", name, "fail (< 5 K)",
                   f"{new['min_pinch_k']:.2f} K (even-grid check {old20['min_pinch_k']:.2f} K)",
                   same and not new["valid"])
            brute = check(cyc, c["sized"], n=1000, phase=False)
            record("T3", name, "= 1,000-point check",
                   f"{new['min_pinch_k']:.3f} vs {brute['min_pinch_k']:.3f} K",
                   abs(new["min_pinch_k"] - brute["min_pinch_k"]) < 0.02)
        else:
            e = evaluate_loop_return(float(o.loop_return_c),
                                     lambda hh, hr, md, _c=c, _o=o: _solver_b(
                                         hh, hr, md, q_in=_c["q_in"], tc=float(_o.tc_c), t_hot=_c["t_hot"]),
                                     c["t_hot"], c["q_target"], DESIGN["t_in_c"], c["c_gas"], GT,
                                     N_TUBES, DUCT_HEIGHT_M)
            ok = (e is not None and math.isclose(e["w_net_kw"], float(o.w_net_kw), rel_tol=1e-9)
                  and abs(e["pinch"]["min_pinch_k"] - float(o.min_pinch_k)) < 1e-9
                  and e["pinch"]["phase_change_fractions"] == [])
            record("T4", name, "unchanged, pass",
                   f"{e['pinch']['min_pinch_k']:.2f} K, {e['w_net_kw']:.2f} kW (was "
                   f"{o.min_pinch_k:.2f} K, {o.w_net_kw:.2f} kW)", ok and e["pinch"]["valid"])
            brute = check(e["cycle"], c["sized"], n=1000, phase=False)
            record("T3", name, "= 1,000-point check",
                   f"{e['pinch']['min_pinch_k']:.3f} vs {brute['min_pinch_k']:.3f} K",
                   abs(e["pinch"]["min_pinch_k"] - brute["min_pinch_k"]) < 0.02)

    # T5 - evaporator area integration, at the frozen ORC+RO design
    o = next(o for name, p, o in designs if name.startswith("orc_ro_current (path A) 85"))
    c = context(85.0, float(o.loop_return_c))
    cyc = cycle_a(float(o.operating_point_c), float(o.tc_c), c["q_in"])
    even = check(cyc, c["sized"], phase=False)
    p = even["profile"]
    n = len(p) - 1
    old_way = sum((even["q_total_w"] / n) / ((p[i]["approach_k"] - p[i + 1]["approach_k"])
                  / math.log(p[i]["approach_k"] / p[i + 1]["approach_k"])
                  if abs(p[i]["approach_k"] - p[i + 1]["approach_k"]) > 1e-9 else p[i]["approach_k"])
                  for i in range(n))
    new_way = area_from_pinch_profile_m2(p, even["q_total_w"], 1.0)
    record("T5", "even grid: new integration = old", "identical",
           f"{new_way / 1e3:,.3f} vs {old_way / 1e3:,.3f} kW/K", math.isclose(new_way, old_way, rel_tol=1e-12))
    new = check(cyc, c["sized"])
    brute = check(cyc, c["sized"], n=1000, phase=False)
    ua_new = area_from_pinch_profile_m2(new["profile"], new["q_total_w"], 1.0)
    ua_brute = area_from_pinch_profile_m2(brute["profile"], brute["q_total_w"], 1.0)
    record("T5", "new grid vs 1,000 slices", "within 1 %",
           f"{ua_new / 1e3:,.1f} vs {ua_brute / 1e3:,.1f} kW/K ({100 * (ua_new / ua_brute - 1):+.2f} %); "
           f"old grid {new_way / 1e3:,.1f}", abs(ua_new / ua_brute - 1) < 0.01)

    width = max(len(r[1]) for r in RESULTS)
    lines = [f"{t:3s}  {case:{width}s}  expected {exp:22s}  got {got:44s}  {ok}" for t, case, exp, got, ok in RESULTS]
    report = "\n".join(lines) + f"\n\n{sum(r[4] == 'PASS' for r in RESULTS)} of {len(RESULTS)} passed\n"
    print(report)
    (HERE / "pinch_check_tests.log").write_text(report)
    return 0 if all(r[4] == "PASS" for r in RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
