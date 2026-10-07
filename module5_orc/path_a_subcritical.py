"""
Path A: subcritical ORC with the evaporating temperature forced low enough
to pass the pinch check. Fully mature technology, and the near-term
baseline. Without the search below, the subcritical cycle's boiling plateau
crosses the loop water inside the evaporator even when both end points look
feasible.

Reuses common/orc_cycle.py's core 4-state subcritical solver unchanged -
this path doesn't need a different cycle model, just (1) the discretised
T-Q pinch check (common/pinch_check.py) against the actual intermediate
loop water stream, and (2) a search that reduces T_evap until the check
passes. Reports both the unconstrained (max-W_net, likely invalid)
result and the constrained (valid) result, so the performance penalty of
staying physically valid is visible rather than hidden.
"""

from common.orc_cycle import solve_orc_cycle
from common.pinch_check import discretized_pinch_check
from common.constants import (
    ORC_WORKING_FLUID, P_WATER_LOOP_PA,
    ORC_TURBINE_ISENTROPIC_EFF, ORC_PUMP_ISENTROPIC_EFF,
)


def _check_cycle_pinch(cycle_result, loop_fluid, p_loop_pa, h_loop_hot, h_loop_return, m_dot_loop,
                        orc_fluid, min_pinch_k, n_segments):
    sp = cycle_result["state_points"]
    return discretized_pinch_check(
        hot_fluid=loop_fluid, p_hot_pa=p_loop_pa, h_hot_in=h_loop_hot, h_hot_out=h_loop_return, m_dot_hot=m_dot_loop,
        cold_fluid=orc_fluid, p_cold_pa=cycle_result["p_evap_pa"], h_cold_in=sp[2]["h"], h_cold_out=sp[3]["h"],
        m_dot_cold=cycle_result["m_dot_wf_kg_s"], n_segments=n_segments, min_pinch_k=min_pinch_k,
    )


def _pack(t_evap_c, cycle, pinch):
    # t_evap_c is returned alongside the cycle dict because
    # cycle["state_points"][3]["t_c"] is T3 = t_evap_c + ORC_SUPERHEAT_C
    # (evaporator exit, after superheat), not t_evap_c itself - reading
    # that back as "T_evap" is an easy mistake, since solve_orc_cycle
    # doesn't echo its t_evap_c input anywhere else in its return dict.
    return {"t_evap_c": t_evap_c, "cycle": cycle, "pinch": pinch}


def solve_path_a(t_evap_unconstrained_c, t_cond_c, q_in_kw, h_loop_hot, h_loop_return, m_dot_loop,
                  fluid=ORC_WORKING_FLUID, p_loop_pa=P_WATER_LOOP_PA, loop_fluid="Water",
                  min_pinch_k=5.0, n_segments=20, t_evap_floor_margin_c=1.0,
                  turbine_eff=ORC_TURBINE_ISENTROPIC_EFF, pump_eff=ORC_PUMP_ISENTROPIC_EFF):
    """
    h_loop_hot/h_loop_return: loop water enthalpy (J/kg) at its hot and
    return temperatures - held fixed during the T_evap search (state 2's
    own enthalpy has a weak secondary dependence on T_evap via the
    pump's compression ratio, but recomputing loop_return per candidate
    would require re-running Module 3's whole water-side sizing loop for
    a second-order effect; not done here).

    Returns dict: 'unconstrained' (t_evap_c + cycle + pinch check at
    t_evap_unconstrained_c, the max-W_net choice, likely invalid) and
    'constrained' (t_evap_c + cycle + pinch check at the highest T_evap
    <= t_evap_unconstrained_c that satisfies the pinch check - None if
    even the floor, t_cond_c + t_evap_floor_margin_c, fails, meaning no
    valid subcritical design exists for these loop conditions at all).
    Each entry's own 't_evap_c' is the reliable way to read back the
    T_evap used - do not infer it from cycle['state_points'][3]['t_c'],
    which is T3 = T_evap + ORC_SUPERHEAT_C (evaporator exit, after
    superheat), not T_evap itself.
    """
    unconstrained_cycle = solve_orc_cycle(t_evap_c=t_evap_unconstrained_c, t_cond_c=t_cond_c, q_in_kw=q_in_kw, fluid=fluid,
                                           turbine_eff=turbine_eff, pump_eff=pump_eff)
    unconstrained_pinch = None
    if unconstrained_cycle is not None:
        unconstrained_pinch = _check_cycle_pinch(unconstrained_cycle, loop_fluid, p_loop_pa, h_loop_hot, h_loop_return,
                                                  m_dot_loop, fluid, min_pinch_k, n_segments)

    unconstrained = _pack(t_evap_unconstrained_c, unconstrained_cycle, unconstrained_pinch)

    if unconstrained_cycle is not None and unconstrained_pinch["valid"]:
        return {"unconstrained": unconstrained, "constrained": unconstrained}

    lo = t_cond_c + t_evap_floor_margin_c
    hi = t_evap_unconstrained_c

    cycle_lo = solve_orc_cycle(t_evap_c=lo, t_cond_c=t_cond_c, q_in_kw=q_in_kw, fluid=fluid,
                               turbine_eff=turbine_eff, pump_eff=pump_eff)
    if cycle_lo is None:
        return {"unconstrained": unconstrained, "constrained": None}
    pinch_lo = _check_cycle_pinch(cycle_lo, loop_fluid, p_loop_pa, h_loop_hot, h_loop_return, m_dot_loop,
                                   fluid, min_pinch_k, n_segments)
    if not pinch_lo["valid"]:
        return {"unconstrained": unconstrained, "constrained": None}

    best_valid = _pack(lo, cycle_lo, pinch_lo)

    for _ in range(40):
        mid = (lo + hi) / 2.0
        cycle_mid = solve_orc_cycle(t_evap_c=mid, t_cond_c=t_cond_c, q_in_kw=q_in_kw, fluid=fluid,
                                    turbine_eff=turbine_eff, pump_eff=pump_eff)
        if cycle_mid is None:
            hi = mid
            continue
        pinch_mid = _check_cycle_pinch(cycle_mid, loop_fluid, p_loop_pa, h_loop_hot, h_loop_return, m_dot_loop,
                                        fluid, min_pinch_k, n_segments)
        if pinch_mid["valid"]:
            best_valid = _pack(mid, cycle_mid, pinch_mid)
            lo = mid
        else:
            hi = mid

    return {"unconstrained": unconstrained, "constrained": best_valid}
