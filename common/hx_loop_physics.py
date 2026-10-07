"""
Flue-gas exchanger geometry and gas-side transport: the duct's tube layout
and free-flow area, and the gas-side heat transfer coefficient, overall U
for each alloy (316L and 904L), density, velocity and volumetric flow at a
record's own flow conditions.

The gas-side Nusselt number is the Zukauskas (1972) staggered tube-bank
correlation, Nu = 0.35 Re^0.6 Pr^0.36.
"""

import numpy as np

from common.constants import (
    N_GAS_TRAINS, TUBE_DO_M, DUCT_HEIGHT_M, TUBE_WALL_M,
    K_316L, K_904L, K_GAS_FLUE, MU_GAS_FLUE, PRANDTL_GAS,
    FOULING_RESISTANCE_BASELINE, FOULING_DUST_NORMALIZER,
    ZUKAUSKAS_C, ZUKAUSKAS_M, ZUKAUSKAS_N,
    P_AMBIENT_KPA,
)


def duct_geometry(duct_width_m, pitch_ratio, tube_do_m=TUBE_DO_M):
    """Tubes per row across the duct, and the minimum (transverse) free-flow area."""
    s_t = pitch_ratio * tube_do_m
    n_tubes_per_row = int(np.floor(duct_width_m / s_t))
    a_flow_minimum = DUCT_HEIGHT_M * (duct_width_m - (n_tubes_per_row * tube_do_m))
    return n_tubes_per_row, a_flow_minimum


def gas_side_heat_transfer(t_in, m_dot_fg_total, p_gauge_kpa, dust_burden, a_flow_minimum):
    """
    Gas-side transport properties and per-alloy overall heat transfer
    coefficient (U) at the record's actual flow conditions, for a FIXED
    duct and tube geometry. Depends only on the flow state (mass flow,
    pressure, dust loading), not on target temperature or area, so the same
    function characterises the design record (once, to size UA and area)
    and rates every other record against that fixed geometry
    (common/ntu_rating.py). Works element-wise on arrays.

    The gas density uses dry air's gas constant (287.05 J/kg-K). The flue
    gas's own is about 280.5, so the gas is about 2.3 % denser than modelled
    and the fan power per row about 4.5 % lower: a conservative simplification.
    The water-side coefficient is a fixed 4,000 W/m2-K; the gas side is about
    95 % of the total resistance, so U is insensitive to it.
    """
    rho_gas = ((P_AMBIENT_KPA + p_gauge_kpa) * 1000.0) / (287.05 * (t_in + 273.15))
    m_dot_fg_duct = m_dot_fg_total / N_GAS_TRAINS
    volumetric_flow_m3s_total = m_dot_fg_total / rho_gas
    volumetric_flow_m3s_duct = volumetric_flow_m3s_total / N_GAS_TRAINS
    v_max = m_dot_fg_duct / (rho_gas * a_flow_minimum)
    re_max = (rho_gas * v_max * TUBE_DO_M) / MU_GAS_FLUE

    nusselt_gas = ZUKAUSKAS_C * (re_max ** ZUKAUSKAS_M) * (PRANDTL_GAS ** ZUKAUSKAS_N)
    h_gas_side = (nusselt_gas * K_GAS_FLUE) / TUBE_DO_M
    h_water_side = 4000.0

    r_fouling = FOULING_RESISTANCE_BASELINE * (dust_burden / FOULING_DUST_NORMALIZER)
    r_convective_terms = (1.0 / h_gas_side) + (1.0 / h_water_side) + r_fouling

    u_316 = 1.0 / (r_convective_terms + (TUBE_WALL_M / K_316L))
    u_904 = 1.0 / (r_convective_terms + (TUBE_WALL_M / K_904L))

    return {
        "rho_gas": rho_gas,
        "volumetric_flow_m3s_total": volumetric_flow_m3s_total,
        "volumetric_flow_m3s_duct": volumetric_flow_m3s_duct,
        "v_max": v_max,
        "re_max": re_max,
        "u_316": u_316,
        "u_904": u_904,
    }
