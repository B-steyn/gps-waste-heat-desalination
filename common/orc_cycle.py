"""
Module 5: Organic Rankine Cycle solver. Four-component subcritical cycle
(evaporator, expander, condenser, pump), without a recuperator.

State points:
  1 = pump inlet = condenser outlet = saturated liquid at Tc
  2 = pump outlet = evaporator inlet = compressed liquid
  3 = evaporator outlet = turbine inlet = superheated vapor at T_evap + superheat
  4 = turbine outlet = condenser inlet

This module is deliberately architecture-agnostic: it takes T_evap, Tc,
and Q_in as plain inputs and solves the cycle. How T_evap and Tc get set
for a given architecture (loop-pinch-derived T_evap, seawater-driven Tc
for ORC+RO, MED-compatible Tc for ORC+MED) is the caller's job: see
module5_orc/path_a_subcritical.py and path_b_transcritical.py.
"""

import CoolProp.CoolProp as CP

from common.constants import (
    ORC_WORKING_FLUID, ORC_TURBINE_ISENTROPIC_EFF, ORC_PUMP_ISENTROPIC_EFF,
    ORC_GENERATOR_EFF, ORC_SUPERHEAT_C, ORC_PARASITIC_FRACTION_OF_GROSS,
)


def solve_orc_cycle(t_evap_c, t_cond_c, q_in_kw, fluid=ORC_WORKING_FLUID,
                     turbine_eff=ORC_TURBINE_ISENTROPIC_EFF, pump_eff=ORC_PUMP_ISENTROPIC_EFF,
                     gen_eff=ORC_GENERATOR_EFF, superheat_c=ORC_SUPERHEAT_C,
                     parasitic_fraction=ORC_PARASITIC_FRACTION_OF_GROSS):
    """
    Solves a 4-component subcritical ORC cycle for the given evaporation
    and condensation temperatures, sizing the working-fluid mass flow so
    the evaporator duty matches q_in_kw (heat delivered by the
    intermediate loop).

    Returns None if t_evap_c <= t_cond_c (no temperature lift, so no
    feasible cycle, e.g. ORC+MED at the deeper tracks) or if q_in_kw <= 0.

    Returns a dict: w_turbine_kw, w_pump_kw, w_parasitic_kw, w_net_kw,
    q_cond_kw, eta_th_pct (= (W_turb - W_pump)/Q_in, before generator
    efficiency), m_dot_wf_kg_s, and the four state points' enthalpy and
    entropy for the exergy balances.

    w_net_kw = gen_eff*W_turbine - W_pump - W_parasitic, where W_parasitic
    covers the auxiliaries (cooling system, controls), distinct from the
    working-fluid pump.
    """
    if q_in_kw <= 0 or t_evap_c <= t_cond_c:
        return None

    t_evap_k = t_evap_c + 273.15
    t_cond_k = t_cond_c + 273.15
    t3_k = t_evap_k + superheat_c

    # state 1: saturated liquid at Tc
    p_cond = CP.PropsSI("P", "T", t_cond_k, "Q", 0, fluid)
    h1 = CP.PropsSI("H", "T", t_cond_k, "Q", 0, fluid)
    s1 = CP.PropsSI("S", "T", t_cond_k, "Q", 0, fluid)

    # state 2: compressed liquid at evaporator pressure, after the pump
    p_evap = CP.PropsSI("P", "T", t_evap_k, "Q", 0, fluid)
    h2s = CP.PropsSI("H", "P", p_evap, "S", s1, fluid)
    h2 = h1 + (h2s - h1) / pump_eff
    s2 = CP.PropsSI("S", "P", p_evap, "H", h2, fluid)

    # state 3: superheated vapor at evaporator pressure, turbine inlet
    h3 = CP.PropsSI("H", "T", t3_k, "P", p_evap, fluid)
    s3 = CP.PropsSI("S", "T", t3_k, "P", p_evap, fluid)

    # state 4: actual turbine outlet at condenser pressure
    h4s = CP.PropsSI("H", "P", p_cond, "S", s3, fluid)
    h4 = h3 - turbine_eff * (h3 - h4s)
    s4 = CP.PropsSI("S", "P", p_cond, "H", h4, fluid)

    # size mass flow so evaporator duty matches the heat the loop delivers
    delta_h_evap = (h3 - h2) / 1000.0  # kJ/kg
    if delta_h_evap <= 0:
        return None
    m_dot_wf = q_in_kw / delta_h_evap

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
        "p_evap_pa": p_evap,
        "p_cond_pa": p_cond,
        "state_points": {
            1: {"h": h1, "s": s1, "t_c": t_cond_c},
            2: {"h": h2, "s": s2, "t_c": CP.PropsSI("T", "P", p_evap, "H", h2, fluid) - 273.15},
            3: {"h": h3, "s": s3, "t_c": t3_k - 273.15},
            4: {"h": h4, "s": s4, "t_c": CP.PropsSI("T", "P", p_cond, "H", h4, fluid) - 273.15},
        },
    }
