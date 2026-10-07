"""
Path B: "transcritical" ORC. An emerging, medium-term option that removes
the isothermal boiling plateau (which forces path A's evaporating
temperature down to within 2-3 K of the condenser) by holding the working
fluid above its critical PRESSURE throughout heat addition, so there is no
two-phase region to cross while it is heated.

Working fluid: R1234ze(E) (Tcrit=109.4C, Pcrit=3.635MPa) - already
anticipated in common/fluid_properties.py's ThermoCore as
`self.ORC_TRANS`. Chosen over R245fa (Tcrit=153.9C, far above anything
this loop reaches) and CO2 (Tcrit=31.0C, below every Tc this model uses,
so it could not condense subcritically). R1234ze(E)'s critical temperature
sits above the loop's hot end (75-95C across tracks), so state 3 sits below
the steep pseudocritical-Cp zone near Tcrit while the pressure is
supercritical, which gives a smooth heating curve.

LIMITATION. Because the loop cannot heat the fluid above its critical
temperature, state 3 (74-90 C, 20-35 K below Tcrit) is a dense, liquid-like
state, and the expansion ends TWO-PHASE (exhaust quality about 0.28-0.43).
This solver applies the dry-vapour turbine efficiency (0.80); path C's rule
for a two-phase expansion is 0.60. At 0.60, path B's net output at 90-100 C
falls to about -107 to +39 kW. Path B is therefore in effect a flash cycle,
its output here is overstated, and it is not a working alternative
(validation/independent_checks.py, section T5b).

Unlike Path A (which reuses orc_cycle.py's solver unchanged - same
cycle, just a constrained T_evap), this path needs its own cycle solver:
there is no saturation temperature/boiling point to derive state 3 from,
so the turbine inlet state is defined directly by (P2, T3), not by a
T_evap + superheat pair.
"""

import CoolProp.CoolProp as CP

from common.pinch_check import discretized_pinch_check
from common.constants import (
    P_WATER_LOOP_PA, ORC_TURBINE_ISENTROPIC_EFF, ORC_PUMP_ISENTROPIC_EFF,
    ORC_GENERATOR_EFF, ORC_PARASITIC_FRACTION_OF_GROSS,
)

ORC_TRANS_FLUID = "R1234ze(E)"
TRANS_PR_OVER_PC = 1.05  # P2 = 1.05 * Pcrit, within the usual 1.01-1.3x range


def solve_transcritical_cycle(t3_c, t_cond_c, q_in_kw, fluid=ORC_TRANS_FLUID, pr_over_pc=TRANS_PR_OVER_PC,
                               turbine_eff=ORC_TURBINE_ISENTROPIC_EFF, pump_eff=ORC_PUMP_ISENTROPIC_EFF,
                               gen_eff=ORC_GENERATOR_EFF, parasitic_fraction=ORC_PARASITIC_FRACTION_OF_GROSS):
    """
    State points:
      1 = pump inlet = condenser outlet = saturated liquid at Tc (must be
          subcritical: t_cond_c < Tcrit, or there is no liquid to pump)
      2 = pump outlet = heater inlet, at P2 = pr_over_pc * Pcrit
      3 = heater outlet = turbine inlet, at P2 and the chosen t3_c
          (continuous single-phase heating, 2->3 - no boiling plateau
          since P2 > Pcrit throughout)
      4 = turbine outlet = condenser inlet, at Pcond

    Returns None if infeasible (t_cond_c >= Tcrit, t3_c doesn't give a
    real temperature/enthalpy rise above state 2, or q_in_kw <= 0) -
    same calling convention as orc_cycle.solve_orc_cycle.
    """
    if q_in_kw <= 0:
        return None

    t_crit_c = CP.PropsSI("Tcrit", fluid) - 273.15
    if t_cond_c >= t_crit_c:
        return None

    p_crit = CP.PropsSI("Pcrit", fluid)
    p_high = pr_over_pc * p_crit
    t_cond_k = t_cond_c + 273.15
    t3_k = t3_c + 273.15

    # state 1: saturated liquid at Tc
    p_cond = CP.PropsSI("P", "T", t_cond_k, "Q", 0, fluid)
    h1 = CP.PropsSI("H", "T", t_cond_k, "Q", 0, fluid)
    s1 = CP.PropsSI("S", "T", t_cond_k, "Q", 0, fluid)

    # state 2: compressed liquid at P2 (supercritical pressure), after the pump
    h2s = CP.PropsSI("H", "P", p_high, "S", s1, fluid)
    h2 = h1 + (h2s - h1) / pump_eff
    s2 = CP.PropsSI("S", "P", p_high, "H", h2, fluid)
    t2_c = CP.PropsSI("T", "P", p_high, "H", h2, fluid) - 273.15
    if t3_c <= t2_c:
        return None

    # state 3: at P2, chosen T3 - continuous single-phase heating from state 2
    h3 = CP.PropsSI("H", "T", t3_k, "P", p_high, fluid)
    s3 = CP.PropsSI("S", "T", t3_k, "P", p_high, fluid)
    if h3 <= h2:
        return None

    # state 4: actual turbine outlet at condenser pressure
    h4s = CP.PropsSI("H", "P", p_cond, "S", s3, fluid)
    h4 = h3 - turbine_eff * (h3 - h4s)
    s4 = CP.PropsSI("S", "P", p_cond, "H", h4, fluid)

    delta_h_heat = (h3 - h2) / 1000.0  # kJ/kg
    if delta_h_heat <= 0:
        return None
    m_dot_wf = q_in_kw / delta_h_heat

    w_turbine_kw = m_dot_wf * (h3 - h4) / 1000.0
    w_pump_kw = m_dot_wf * (h2 - h1) / 1000.0
    w_parasitic_kw = parasitic_fraction * w_turbine_kw
    w_net_kw = gen_eff * w_turbine_kw - w_pump_kw - w_parasitic_kw
    q_cond_kw = m_dot_wf * (h4 - h1) / 1000.0

    eta_th_pct = ((w_turbine_kw - w_pump_kw) / q_in_kw) * 100.0

    return {
        "m_dot_wf_kg_s": m_dot_wf,
        "w_turbine_kw": w_turbine_kw,
        "w_pump_kw": w_pump_kw,
        "w_parasitic_kw": w_parasitic_kw,
        "w_net_kw": w_net_kw,
        "q_cond_kw": q_cond_kw,
        "eta_th_pct": eta_th_pct,
        "t3_c": t3_c,
        "t2_c": t2_c,
        "p_high_pa": p_high,
        "p_cond_pa": p_cond,
        "reduced_pressure": pr_over_pc,
        "state_points": {
            1: {"h": h1, "s": s1, "t_c": t_cond_c},
            2: {"h": h2, "s": s2, "t_c": t2_c},
            3: {"h": h3, "s": s3, "t_c": t3_c},
            4: {"h": h4, "s": s4, "t_c": CP.PropsSI("T", "P", p_cond, "H", h4, fluid) - 273.15},
        },
    }


def _check_cycle_pinch(cycle_result, loop_fluid, p_loop_pa, h_loop_hot, h_loop_return, m_dot_loop,
                        orc_fluid, min_pinch_k, n_segments):
    sp = cycle_result["state_points"]
    return discretized_pinch_check(
        hot_fluid=loop_fluid, p_hot_pa=p_loop_pa, h_hot_in=h_loop_hot, h_hot_out=h_loop_return, m_dot_hot=m_dot_loop,
        cold_fluid=orc_fluid, p_cold_pa=cycle_result["p_high_pa"], h_cold_in=sp[2]["h"], h_cold_out=sp[3]["h"],
        m_dot_cold=cycle_result["m_dot_wf_kg_s"], n_segments=n_segments, min_pinch_k=min_pinch_k,
    )


def _pack(t3_c, cycle, pinch):
    return {"t3_c": t3_c, "cycle": cycle, "pinch": pinch}


def solve_path_b(t3_search_low_c, t3_search_high_c, t_cond_c, q_in_kw, h_loop_hot, h_loop_return, m_dot_loop,
                  fluid=ORC_TRANS_FLUID, pr_over_pc=TRANS_PR_OVER_PC, p_loop_pa=P_WATER_LOOP_PA, loop_fluid="Water",
                  min_pinch_k=5.0, n_segments=20, n_scan=40,
                  turbine_eff=ORC_TURBINE_ISENTROPIC_EFF, pump_eff=ORC_PUMP_ISENTROPIC_EFF):
    """
    Unlike Path A (W_net monotonic in T_evap, so "search downward until
    valid" finds the max-W_net valid point automatically), Path B's
    W_net-vs-T3 relationship isn't assumed monotonic: raising T3 both
    increases specific turbine work AND decreases m_dot_wf (same fixed
    Q_in spread over a bigger enthalpy rise), so the net direction isn't
    obvious a priori. This scans T3 across [t3_search_low_c,
    t3_search_high_c] at n_scan points, evaluates W_net and pinch
    validity at each, and returns the actual max-W_net point among the
    valid ones (not just the first/highest valid point found).

    Returns dict: 'unconstrained' (t3_c + cycle + pinch at the scan's
    highest-W_net point overall, regardless of validity) and
    'constrained' (t3_c + cycle + pinch at the highest-W_net point among
    valid ones - None if no scanned point is valid). Also returns 'scan'
    (the full list of {t3_c, cycle, pinch} for every point, so the
    W_net-vs-T3 shape can be inspected/plotted).
    """
    scan = []
    for i in range(n_scan + 1):
        t3_c = t3_search_low_c + (t3_search_high_c - t3_search_low_c) * i / n_scan
        cycle = solve_transcritical_cycle(t3_c=t3_c, t_cond_c=t_cond_c, q_in_kw=q_in_kw, fluid=fluid,
                                           pr_over_pc=pr_over_pc, turbine_eff=turbine_eff, pump_eff=pump_eff)
        if cycle is None:
            continue
        pinch = _check_cycle_pinch(cycle, loop_fluid, p_loop_pa, h_loop_hot, h_loop_return, m_dot_loop,
                                    fluid, min_pinch_k, n_segments)
        scan.append(_pack(t3_c, cycle, pinch))

    if not scan:
        return {"unconstrained": None, "constrained": None, "scan": scan}

    unconstrained = max(scan, key=lambda entry: entry["cycle"]["w_net_kw"])
    valid_points = [entry for entry in scan if entry["pinch"]["valid"]]
    constrained = max(valid_points, key=lambda entry: entry["cycle"]["w_net_kw"]) if valid_points else None

    return {"unconstrained": unconstrained, "constrained": constrained, "scan": scan}
