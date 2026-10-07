"""
The FGCHE sizing chain: the deterministic sequence that turns a chosen
loop return temperature into physical hardware and its parasitic loads.

    loop_return_c -> m_dot_loop -> capacity rate -> UA -> area
                  -> tube rows -> gas-side dP -> ID fan power

One implementation, shared by Module 3 and the loop optimiser, so the two
cannot drift apart.

Convention: `q_target_kw`, `c_water_total` and the returned areas are TOTAL
(both ducts); `c_gas_per_duct` and everything inside `gas_transport` are PER
DUCT. The UA solve is per duct. Mixing the two would silently push the
target effectiveness above 1.
"""

import numpy as np
import CoolProp.CoolProp as CP

from common.ntu_rating import solve_required_ua
from common.constants import (
    N_TRAINS_FITTED, P_WATER_LOOP_PA, TUBE_DO_M, TUBE_ROW_DP_FACTOR, FAN_EFFICIENCY,
    RHO_LIQUID_WATER, PIPE_LENGTH_M, PUMP_EFFICIENCY,
    PIPE_FRICTION_PAM,
)


def rows_for_area(area_total_m2, n_tubes_per_row, duct_height_m):
    """Tube rows per duct for a given TOTAL (both-duct) surface area."""
    return float(np.ceil((area_total_m2 / N_TRAINS_FITTED) /
                          (n_tubes_per_row * np.pi * TUBE_DO_M * duct_height_m)))


def size_fgche(loop_return_c, t_water_hot_c, q_target_kw, t_gas_in_c,
                c_gas_per_duct_kw_per_k, gas_transport, n_tubes_per_row, duct_height_m,
                p_loop_pa=P_WATER_LOOP_PA):
    """
    Sizes the exchanger at one design point. Returns None if the loop
    has no temperature lift available (return at or above the hot side).

    Areas are returned for both candidate alloys because they differ
    only through the overall heat transfer coefficient U - the same UA
    requires more area in the less conductive alloy.
    """
    if loop_return_c >= t_water_hot_c:
        return None

    h_hot = CP.PropsSI("H", "T", t_water_hot_c + 273.15, "P", p_loop_pa, "Water")
    h_ret = CP.PropsSI("H", "T", loop_return_c + 273.15, "P", p_loop_pa, "Water")

    m_dot_loop = q_target_kw / ((h_hot - h_ret) / 1000.0)
    c_water_total = q_target_kw / (t_water_hot_c - loop_return_c)

    ua_per_duct = solve_required_ua(q_target_kw / N_TRAINS_FITTED, c_gas_per_duct_kw_per_k,
                                     c_water_total / N_TRAINS_FITTED, t_gas_in_c, loop_return_c)

    # area[m2] = UA[kW/K] * 1000 / U[W/m2-K], x2 for both ducts
    area_316 = N_TRAINS_FITTED * (ua_per_duct * 1000.0 / gas_transport["u_316"])
    area_904 = N_TRAINS_FITTED * (ua_per_duct * 1000.0 / gas_transport["u_904"])

    return {
        "loop_return_c": loop_return_c,
        "h_loop_hot": h_hot,
        "h_loop_return": h_ret,
        "m_dot_loop_kg_s": m_dot_loop,
        "c_water_total_kw_per_k": c_water_total,
        "ua_per_duct_kw_per_k": ua_per_duct,
        "area_316_m2": area_316,
        "area_904_m2": area_904,
        "rows_316": rows_for_area(area_316, n_tubes_per_row, duct_height_m),
        "rows_904": rows_for_area(area_904, n_tubes_per_row, duct_height_m),
    }


def gas_side_fan_power(rows, gas_transport):
    """
    Gas-side pressure drop across the tube bank and the resulting ID fan
    power, for BOTH ducts.

    Called with the DESIGN record's gas transport to size a design and
    check it against the fan's limit, and with each operating record's own
    gas transport for the annual energy (fan_margin.operating_record_factors):
    fan power tracks the actual flow, as m^3/rho^2.

    TUBE_ROW_DP_FACTOR is effectively an Euler number per row and is
    uncited; published staggered tube-bank values tend to run higher, which
    would raise the fan power. Fan power scales with v_max^2, so roughly
    with 1/duct_area^2, which is why the duct dimensions matter so much.
    """
    dp_pa = rows * TUBE_ROW_DP_FACTOR * (0.5 * gas_transport["rho_gas"] * gas_transport["v_max"] ** 2)
    fan_kw = N_TRAINS_FITTED * ((gas_transport["volumetric_flow_m3s_duct"] * dp_pa / FAN_EFFICIENCY) / 1000.0)
    return {"dp_pa": dp_pa, "fan_kw": fan_kw}


def loop_pump_power_kw(m_dot_loop_kg_s):
    """
    Circulating pump power for the intermediate loop.

    The loop is CLOSED and pressurised (P_WATER_LOOP_PA), so the pump works
    against friction only: elevation climbed on the supply leg is recovered
    on the return leg. A static-lift term (rho g h) would size a fill pump,
    not a circulator.

    LIMITATION: exchanger tube-side pressure drops (FGCHE, ORC evaporator,
    preheater, MED heating section) are not modelled, because tube-side
    flow is not resolved (h_water is a fixed 4000 W/m2K). They are of the
    same order as the pipe friction. The pump is under 0.3 % of LCOW, so
    neither term can move a reported result.
    """
    dp_pump_pa = PIPE_LENGTH_M * PIPE_FRICTION_PAM
    return (m_dot_loop_kg_s * dp_pump_pa / RHO_LIQUID_WATER / PUMP_EFFICIENCY) / 1000.0
