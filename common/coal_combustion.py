"""
Coal combustion stoichiometry, acid dew point (Okkes correlation), and
cold-end corrosion risk. The excess-air derivation and theoretical O2
demand are hand-verified against a mass balance.

Module 1 evaluates it for every operating record. Modules 2b and 3 and the
design record (module5_orc/design_cases.py) call the same functions, so
every module works from one combustion model.
"""

import numpy as np

from common.constants import (
    C_DAF, H_DAF, N_DAF, O_DAF, S_DAF,
    MOIST_AD, ASH_AD, TOTAL_MOIST_AR,
    CONVERSION_SO2_TO_SO3, ASH_CAPTURE_EFF,
)


def calculate_dynamic_adp(c_daf=C_DAF, h_daf=H_DAF, n_daf=N_DAF, o_daf=O_DAF, s_daf=S_DAF,
                           moist_ad=MOIST_AD, ash_ad=ASH_AD, total_moist_ar=TOTAL_MOIST_AR):
    """
    Translates air-dried proximate and DAF ultimate elements into an
    as-received basis to calculate the continuous chemical baseline
    properties of the coal fuel blend. Returns the per-fuel mole
    quantities needed by combustion_mole_balance() for every row.
    """
    fraction_dry_from_ad = 1.0 - (moist_ad / 100.0)
    ash_dry_basis = (ash_ad / 100.0) / fraction_dry_from_ad
    ash_ar = ash_dry_basis * (1.0 - total_moist_ar)
    x_daf_ar = 1.0 - total_moist_ar - ash_ar

    c_ar = c_daf * x_daf_ar
    h_ar = h_daf * x_daf_ar
    o_ar = o_daf * x_daf_ar
    n_ar = n_daf * x_daf_ar
    s_ar = s_daf * x_daf_ar

    n_co2 = c_ar / 12.011
    n_h2o_total = (h_ar / 2.016) + (total_moist_ar / 18.015)

    n_so2_initial = s_ar / 32.06
    n_so3_formed = n_so2_initial * CONVERSION_SO2_TO_SO3
    n_so3_gas = n_so3_formed * (1.0 - ASH_CAPTURE_EFF)   # the rest is captured by ash
    # Gaseous SO2 is what was NOT converted to SO3. SO3 captured by the
    # alkaline fly ash leaves the gas as sulphate, so it is not returned to
    # the gas as SO2 and the sulphur balance closes.
    n_so2_gas = n_so2_initial - n_so3_formed
    n_n2_fuel = n_ar / 28.013

    # + half a mole of O2 per mole of SO3 formed (S -> SO2 -> SO3)
    n_o2_theo = n_co2 + (h_ar / 4.032) + n_so2_initial + 0.5 * n_so3_formed - (o_ar / 31.998)
    n_base = n_co2 + n_h2o_total + n_so2_gas + n_so3_gas + n_n2_fuel - n_o2_theo

    return n_co2, n_h2o_total, n_so2_gas, n_so3_gas, n_n2_fuel, n_o2_theo, n_base


def combustion_mole_balance(o2_meas_pct, coal_moles, p_gauge_kpa, p_ambient_kpa):
    """
    Given measured stack O2 (%) and the fixed per-fuel mole quantities from
    calculate_dynamic_adp(), returns dynamic excess-air lambda, flue gas
    H2O/SO3 mole fractions, and the Okkes acid dew point (°C) for one row.

    Returns None if the row is non-physical (O2 >= 21%, i.e. no combustion).
    """
    n_co2, n_h2o_total, n_so2_gas, n_so3_gas, n_n2_fuel, n_o2_theo, n_base = coal_moles

    if o2_meas_pct >= 21.0:
        return None

    y_o2_meas = o2_meas_pct / 100.0
    lamb = (n_o2_theo + y_o2_meas * n_base) / (n_o2_theo * (1.0 - (o2_meas_pct / 21.0)))

    n_o2_exhaust = (lamb - 1.0) * n_o2_theo
    n_n2_air = lamb * n_o2_theo * (79.0 / 21.0)
    n_tot = n_co2 + n_h2o_total + n_so2_gas + n_so3_gas + n_o2_exhaust + n_n2_air + n_n2_fuel

    y_h2o = n_h2o_total / n_tot
    y_so3 = n_so3_gas / n_tot
    y_co2 = n_co2 / n_tot
    y_o2 = n_o2_exhaust / n_tot
    y_n2 = (n_n2_air + n_n2_fuel) / n_tot

    p_tot_atm = (p_ambient_kpa + p_gauge_kpa) / 101.325
    ph2o_atm = y_h2o * p_tot_atm
    pso3_atm = y_so3 * p_tot_atm

    if pso3_atm <= 0:
        t_adp = 45.0
    else:
        # Okkes correlation, as published (the +8.0 constant checked
        # against several sources).
        t_adp = (203.25 + 27.6 * np.log10(ph2o_atm) + 10.83 * np.log10(pso3_atm)
                  + 1.06 * (np.log10(pso3_atm) + 8.0) ** 2.19)

    return {
        "lambda": lamb,
        "y_h2o": y_h2o,
        "y_so3": y_so3,
        "y_co2": y_co2,
        "y_o2": y_o2,
        "y_n2": y_n2,
        "so3_ppm": y_so3 * 1e6,
        "adp_c": t_adp,
    }


def calculate_corrosion_index(t_adp, t_surface):
    """
    Empirical asymmetric curve modelling relative cold-end structural
    degradation risk from sulfuric acid condensation. Returns a normalized
    risk scale from 0.0 (safe) to 1.0 (peak corrosion).

    Peak location (22 C below ADP) is backed by: E. Land, "The theory of
    acid deposition and its application to the dew point meter," Journal
    of the Institute of Fuel, June 1977 (cited via Tetra Engineering's
    HRSG corrosion whitepaper) - peak deposition rate reported at 20-30 C
    below dew point for particulate-poor gas. The Gaussian shape itself is
    an approximation (no published equation found for the curve shape).

    Limitation: this tap point is post-fabric-filter, so most particulate
    that would otherwise reduce structural deposition severity (per Land)
    has already been removed upstream. The SO3-reduction effect (ash
    capture) and the loss-of-particulate-scavenging effect pull in
    opposite directions and are not both captured by this index.

    Scope note: sulfuric acid dew point corrosion (Okkes) is the only
    corrosion mechanism modelled. Nitric acid corrosion is not: a fixed
    NOx-to-HNO3 conversion could not be justified chemically.
    """
    delta_t_dep = t_adp - t_surface
    if delta_t_dep <= 0:
        return 0.0
    index_val = np.exp(-((delta_t_dep - 22.0) ** 2) / (2.0 * (7.0 ** 2)))
    return float(index_val)
