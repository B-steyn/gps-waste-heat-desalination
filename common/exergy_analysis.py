"""
Component-level exergy destruction (Gouy-Stodola theorem):
E_D,k = T0 * S_gen,k, reported both in kW and as a fraction of the exergy
entering.

Module 2 gives the idealised exergy content of the recovered heat band (a
reversible-recovery upper bound). These functions give the real
destruction once Module 3 supplies two-sided state points (flue gas in and
out, loop water in and out).

Scope: the flue-gas exchanger and intermediate loop with their fan and pump
work, and the ORC turbine, pump and evaporator (state points from
solve_orc_cycle). The ORC condenser is not treated here: in these
cogeneration architectures its heat is a useful product (RO preheat or MED
heat), not waste rejected to the dead state, so its accounting belongs with
the plant that receives it. The design-record balances of the reported
designs, including the condenser, are built in analysis/energy_tables.py.
"""

import numpy as np

from common.constants import T_AMBIENT_CELSIUS, P_AMBIENT_KPA


def gas_exergy_and_entropy_change(m_dot_fg, cp_flue_gas, t_start_c, t_end_c, t0_celsius=T_AMBIENT_CELSIUS):
    """
    Ideal-gas flow exergy and entropy change of a flue-gas stream cooling
    (or warming) from t_start_c to t_end_c, using the same constant-cp-
    over-the-span approximation already used throughout the model
    (algebraically the ideal-gas / constant-Cp form of psi = (h-h0) -
    T0(s-s0)). Returns (delta_ex_kw, delta_s_kw_per_k) - delta_s is
    negative for a cooling span.
    """
    t0_k = t0_celsius + 273.15
    t_start_k = t_start_c + 273.15
    t_end_k = t_end_c + 273.15

    delta_s_kw_per_k = m_dot_fg * cp_flue_gas * np.log(t_end_k / t_start_k)
    # flow exergy relative to the dead state, evaluated at t_start_c,
    # minus the same at t_end_c - i.e. the exergy given up over the span
    ex_at_start = m_dot_fg * cp_flue_gas * ((t_start_c - t0_celsius) - t0_k * np.log(t_start_k / t0_k))
    ex_at_end = m_dot_fg * cp_flue_gas * ((t_end_c - t0_celsius) - t0_k * np.log(t_end_k / t0_k))
    delta_ex_kw = ex_at_start - ex_at_end

    return delta_ex_kw, delta_s_kw_per_k, ex_at_end


def water_exergy_gain(thermo, m_dot_water, t_in_c, t_out_c, p_water_pa, t0_celsius=T_AMBIENT_CELSIUS):
    """
    Real (CoolProp state-point) exergy and entropy gain of the
    intermediate loop water as it heats from t_in_c to t_out_c at fixed
    loop pressure. Returns (ex_gain_kw, delta_s_kw_per_k, q_absorbed_kw).
    """
    t0_k = t0_celsius + 273.15

    props_in = thermo.get_fluid_properties(thermo.WATER, "T", t_in_c + 273.15, "P", p_water_pa)
    props_out = thermo.get_fluid_properties(thermo.WATER, "T", t_out_c + 273.15, "P", p_water_pa)

    delta_h_kw = m_dot_water * (props_out["h"] - props_in["h"]) / 1000.0
    delta_s_kw_per_k = m_dot_water * (props_out["s"] - props_in["s"]) / 1000.0
    ex_gain_kw = delta_h_kw - t0_k * delta_s_kw_per_k

    return ex_gain_kw, delta_s_kw_per_k, delta_h_kw


def hx_exergy_destruction(m_dot_fg, cp_flue_gas, t_in_c, t_target_c, m_dot_water, t_water_in_c, t_water_out_c,
                           q_shell_loss_kw, thermo, p_water_pa, t0_celsius=T_AMBIENT_CELSIUS):
    """
    Component-level exergy destruction across the FGCHE/intermediate-loop
    heat exchanger, treated as one control volume including the shell
    heat leak (modelled as heat rejected directly to the T0 environment,
    per HX_SHELL_EFFICIENCY < 1).

    Control-volume entropy balance (steady state, both streams plus the
    shell-loss heat crossing the boundary to the T0 environment):
    S_gen = (s_out - s_in) summed over both streams - Q_in/T_boundary.
    The gas stream's (s_out - s_in) is delta_S_gas, already negative
    since it cools; the water stream's is delta_S_water, positive since
    it heats; the shell-loss heat leaves the CV (not enters), so its
    contribution is +Q_shell_loss/T0, not negative.

    E_D = T0 * S_gen  (Gouy-Stodola)

    Also returns the terms of the exergy balance across the gas side of
    the HX - exergy given up by the gas (T_in -> T_target) splits into
    what the water actually gained plus what was destroyed:
        exergy_given_up_by_gas_kw = exergy_gain_water_kw + exergy_destroyed_kw
    - as a regression/consistency check, see validation/
    stage1_energy_balance.py. exergy_residual_in_gas_kw is a separate,
    non-balance quantity: how much exergy the gas stream still carries
    when it leaves the HX at T_target (relative to dead state) - i.e.
    what's left on the table, still available to a downstream tap.
    """
    t0_k = t0_celsius + 273.15

    exergy_given_up_by_gas, delta_s_gas, ex_residual_in_exiting_gas = gas_exergy_and_entropy_change(
        m_dot_fg, cp_flue_gas, t_in_c, t_target_c, t0_celsius)

    ex_gain_water, delta_s_water, q_absorbed_water = water_exergy_gain(
        thermo, m_dot_water, t_water_in_c, t_water_out_c, p_water_pa, t0_celsius)

    s_gen_kw_per_k = delta_s_gas + delta_s_water + (q_shell_loss_kw / t0_k)
    e_destroyed_kw = t0_k * s_gen_kw_per_k

    return {
        "exergy_destroyed_kw": e_destroyed_kw,
        "entropy_generated_kw_per_k": s_gen_kw_per_k,
        "exergy_given_up_by_gas_kw": exergy_given_up_by_gas,
        "exergy_residual_in_gas_kw": ex_residual_in_exiting_gas,
        "exergy_gain_water_kw": ex_gain_water,
    }


def parasitic_exergy_destruction(work_kw):
    """
    Exergy destroyed by an ID fan or water pump. Simplifying assumption,
    stated explicitly: nearly all shaft/electrical work put into
    overcoming friction and turbulence in these components degrades
    directly to heat at near-ambient temperature, with negligible useful
    pressure-rise exergy recovered by the working fluid - so the
    parasitic work input is treated as fully destroyed exergy
    (E_D ~= work_kw), rather than modelling the small actual
    isentropic-vs-actual compression exergy split.
    """
    return work_kw
