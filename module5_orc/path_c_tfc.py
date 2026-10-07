"""
Path C: Trilateral Flash Cycle (TFC). Cycle-level functions only: nothing
in run_all.py calls them.

CYCLE LEVEL ONLY - deliberately not integrated into Module 6. A two-phase
expander is research-stage and not bankable, so costing it would imply a
maturity it does not have.

WHY IT IS STILL WORTH BUILDING. Path C was scoped as a thermodynamic upper
bound on the ORC. In this application the ORC's electrical output barely
matters: electricity is about 5 % of LCOW, and the LCOW-optimal design
minimises ORC output. So a better cycle is worth almost nothing, and Path C
answers a sharper question: does the ORC architecture choice matter AT ALL? If
even the theoretically best cycle - no boiling plateau, perfect
temperature glide matching - cannot move LCOW meaningfully, then the
subcritical/transcritical/TFC question is settled for this application,
and that is a stronger conclusion than leaving the upper bound
unevaluated.

THE CYCLE. Heat only to saturated LIQUID, then expand two-phase:
  1: saturated liquid at Tc (pump inlet)
  2: pump outlet at P_high
  3: SATURATED LIQUID at P_high (Q=0) - all heat addition is sensible,
     so there is no boiling plateau at all and the heating curve can
     follow the loop water's cooling curve closely
  4: two-phase expander outlet at P_cond

The absence of a plateau is the whole point: it is the structural
opposite of the boiling plateau that limits path A.
"""

import CoolProp.CoolProp as CP

from common.pinch_check import discretized_pinch_check
from common.constants import (
    P_WATER_LOOP_PA, ORC_PUMP_ISENTROPIC_EFF, ORC_GENERATOR_EFF,
    ORC_PARASITIC_FRACTION_OF_GROSS,
)
from path_b_transcritical import ORC_TRANS_FLUID

# Conservative single-efficiency fallback for two-phase expansion. NEVER use
# dry-vapour expander efficiencies (0.80+) here - that is the single most
# common way TFC studies overstate their results.
TFC_ETA_2PH_CONSERVATIVE = 0.60      # within the 0.55-0.70 band for two-phase expanders
TFC_ETA_2PH_BAND = (0.55, 0.70)


def nozzle_rotor_efficiency(p_cond_pa, x4, fluid=ORC_TRANS_FLUID):
    """
    Lai et al. style nozzle/rotor split:

        eta_nozzle = 0.865 + 0.00175 * rho_v,sat(P_low)
        eta_rotor  = 0.575 + 0.325 * x4

    Returns both and their product. The correlation is reported for the
    specific machines those authors studied; applying it to a different
    fluid and scale is an extrapolation, so the conservative single-value
    fallback is carried alongside and both are reported.
    """
    rho_v = CP.PropsSI("D", "P", p_cond_pa, "Q", 1, fluid)
    eta_nozzle = 0.865 + 0.00175 * rho_v
    eta_rotor = 0.575 + 0.325 * max(min(x4, 1.0), 0.0)
    return {"rho_v_sat": rho_v, "eta_nozzle": eta_nozzle,
            "eta_rotor": eta_rotor, "eta_overall": eta_nozzle * eta_rotor}


def solve_tfc_cycle(t3_c, t_cond_c, q_in_kw, fluid=ORC_TRANS_FLUID,
                     pump_eff=ORC_PUMP_ISENTROPIC_EFF, gen_eff=ORC_GENERATOR_EFF,
                     parasitic_fraction=ORC_PARASITIC_FRACTION_OF_GROSS,
                     expander_model="nozzle_rotor",
                     eta_2ph=TFC_ETA_2PH_CONSERVATIVE):
    """
    `t3_c` is the temperature at the top of the sensible heating, i.e. the
    saturation temperature at P_high. State 3 is saturated LIQUID there -
    the fluid is heated to the bubble point and no further.

    `expander_model`: "nozzle_rotor" applies the correlation above;
    "conservative" applies a single two-phase isentropic efficiency.
    Report both, since the correlation is being extrapolated beyond the
    machines it was fitted to.

    Returns None if infeasible (t3 above critical, no lift, or q_in <= 0).
    """
    if q_in_kw <= 0 or t3_c <= t_cond_c:
        return None
    t_crit_c = CP.PropsSI("Tcrit", fluid) - 273.15
    if t3_c >= t_crit_c:
        return None

    t_cond_k = t_cond_c + 273.15
    t3_k = t3_c + 273.15

    p_cond = CP.PropsSI("P", "T", t_cond_k, "Q", 0, fluid)
    p_high = CP.PropsSI("P", "T", t3_k, "Q", 0, fluid)

    h1 = CP.PropsSI("H", "T", t_cond_k, "Q", 0, fluid)
    s1 = CP.PropsSI("S", "T", t_cond_k, "Q", 0, fluid)

    h2s = CP.PropsSI("H", "P", p_high, "S", s1, fluid)
    h2 = h1 + (h2s - h1) / pump_eff
    s2 = CP.PropsSI("S", "P", p_high, "H", h2, fluid)
    t2_c = CP.PropsSI("T", "P", p_high, "H", h2, fluid) - 273.15

    # state 3: saturated LIQUID at P_high - the defining feature of a TFC
    h3 = CP.PropsSI("H", "P", p_high, "Q", 0, fluid)
    s3 = CP.PropsSI("S", "P", p_high, "Q", 0, fluid)
    if h3 <= h2:
        return None

    # isentropic flash to condenser pressure, landing in the two-phase region
    h4s = CP.PropsSI("H", "P", p_cond, "S", s3, fluid)
    x4s = CP.PropsSI("Q", "P", p_cond, "H", h4s, fluid)

    if expander_model == "nozzle_rotor":
        eff = nozzle_rotor_efficiency(p_cond, x4s, fluid)
        eta_used = eff["eta_overall"]
    else:
        eff = {"eta_overall": eta_2ph, "eta_nozzle": None, "eta_rotor": None,
               "rho_v_sat": None}
        eta_used = eta_2ph

    h4 = h3 - eta_used * (h3 - h4s)
    x4 = CP.PropsSI("Q", "P", p_cond, "H", h4, fluid)
    s4 = CP.PropsSI("S", "P", p_cond, "H", h4, fluid)

    delta_h_heat = (h3 - h2) / 1000.0
    m_dot_wf = q_in_kw / delta_h_heat

    w_exp_kw = m_dot_wf * (h3 - h4) / 1000.0
    w_pump_kw = m_dot_wf * (h2 - h1) / 1000.0
    w_parasitic_kw = parasitic_fraction * w_exp_kw
    w_net_kw = gen_eff * w_exp_kw - w_pump_kw - w_parasitic_kw
    q_cond_kw = m_dot_wf * (h4 - h1) / 1000.0

    return {
        "m_dot_wf_kg_s": m_dot_wf,
        "w_turbine_kw": w_exp_kw, "w_pump_kw": w_pump_kw,
        "w_parasitic_kw": w_parasitic_kw, "w_net_kw": w_net_kw,
        "q_cond_kw": q_cond_kw,
        "eta_th_pct": ((w_exp_kw - w_pump_kw) / q_in_kw) * 100.0,
        "t3_c": t3_c, "t2_c": t2_c,
        "p_high_pa": p_high, "p_cond_pa": p_cond,
        "x4_isentropic": x4s, "x4_actual": x4,
        "expander": eff, "expander_model": expander_model,
        "state_points": {
            1: {"h": h1, "s": s1, "t_c": t_cond_c},
            2: {"h": h2, "s": s2, "t_c": t2_c},
            3: {"h": h3, "s": s3, "t_c": t3_c},
            4: {"h": h4, "s": s4, "t_c": CP.PropsSI("T", "P", p_cond, "H", h4, fluid) - 273.15},
        },
    }


def _check_cycle_pinch(cycle, loop_fluid, p_loop_pa, h_loop_hot, h_loop_return,
                        m_dot_loop, orc_fluid, min_pinch_k, n_segments):
    sp = cycle["state_points"]
    return discretized_pinch_check(
        hot_fluid=loop_fluid, p_hot_pa=p_loop_pa, h_hot_in=h_loop_hot,
        h_hot_out=h_loop_return, m_dot_hot=m_dot_loop,
        cold_fluid=orc_fluid, p_cold_pa=cycle["p_high_pa"],
        h_cold_in=sp[2]["h"], h_cold_out=sp[3]["h"],
        m_dot_cold=cycle["m_dot_wf_kg_s"], n_segments=n_segments,
        min_pinch_k=min_pinch_k)


def solve_path_c(t3_search_low_c, t3_search_high_c, t_cond_c, q_in_kw,
                  h_loop_hot, h_loop_return, m_dot_loop, fluid=ORC_TRANS_FLUID,
                  p_loop_pa=P_WATER_LOOP_PA, loop_fluid="Water",
                  min_pinch_k=5.0, n_segments=20, n_scan=40,
                  expander_model="nozzle_rotor"):
    """
    Scans t3 and returns the max-W_net point among pinch-valid designs.

    The pinch check should be far easier to satisfy than for Path A: TFC
    heating is entirely sensible, so the working fluid's temperature rises
    continuously and can track the loop water's cooling curve instead of
    sitting on a plateau beneath it: the TFC's structural advantage.
    """
    scan = []
    for i in range(n_scan + 1):
        t3 = t3_search_low_c + (t3_search_high_c - t3_search_low_c) * i / n_scan
        cyc = solve_tfc_cycle(t3, t_cond_c, q_in_kw, fluid=fluid,
                               expander_model=expander_model)
        if cyc is None:
            continue
        pinch = _check_cycle_pinch(cyc, loop_fluid, p_loop_pa, h_loop_hot,
                                    h_loop_return, m_dot_loop, fluid,
                                    min_pinch_k, n_segments)
        scan.append({"t3_c": t3, "cycle": cyc, "pinch": pinch})

    if not scan:
        return {"unconstrained": None, "constrained": None, "scan": scan}

    unconstrained = max(scan, key=lambda e: e["cycle"]["w_net_kw"])
    valid = [e for e in scan if e["pinch"]["valid"]]
    constrained = max(valid, key=lambda e: e["cycle"]["w_net_kw"]) if valid else None
    return {"unconstrained": unconstrained, "constrained": constrained, "scan": scan}
