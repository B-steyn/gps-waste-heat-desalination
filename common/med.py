"""
MED cascade relations - one implementation, shared by Module 4a, the
effect-count sweep and Module 6.

The top brine temperature (TBT) is an explicit design input and the effect
count is tied to it: a higher TBT allows more effects (more water per kW of
heat) but forces a warmer loop return, since the loop must stay a pinch
above the TBT along the whole first-effect heater. The effect count is the
lever that trades capital against water.

Uncited parameters are flagged as such.
"""

import numpy as np

MED_EVAP_EFFECTIVENESS = 0.95     # per-effect transfer effectiveness - UNCITED
MED_DELTA_T_PER_EFFECT_C = 3.0    # minimum temperature drop per effect - UNCITED
MED_CONDENSER_APPROACH_C = 5.0    # last-effect condensing temperature above seawater intake
MED_TBT_FLOOR_C = 55.0            # lowest TBT considered - UNCITED
MED_TBT_CEILING_C = 90.0          # upper bound from the project's progress report; MED practice usually
                                  # limits TBT to about 70 C for scaling, so higher TBTs should be read with care


def gor_for_effects(n_effects, effectiveness=MED_EVAP_EFFECTIVENESS):
    """
    GOR = e(1 - e^N) / (1 - e).

    Effect 1 receives q x e; each later effect receives the previous
    effect's vapour duty x e. Verified against Module 4a's iterative
    cascade (5.03 here vs 5.13 there at N = 6; the gap is Module 4a's
    temperature-dependent latent heat).
    """
    e = effectiveness
    return e * (1.0 - e ** n_effects) / (1.0 - e)


def max_feasible_effects(t_tbt_c, t_condenser_c, delta_t_per_effect=MED_DELTA_T_PER_EFFECT_C):
    """Physical cap: the TBT-to-condenser range over the minimum per-effect drop."""
    if t_tbt_c <= t_condenser_c:
        return 0
    return int(np.floor((t_tbt_c - t_condenser_c) / delta_t_per_effect))


def condenser_temperature_c(t_seawater_c, approach_c=MED_CONDENSER_APPROACH_C):
    """Last-effect condensing temperature, set by the seawater it rejects to."""
    return t_seawater_c + approach_c


def latent_heat_kj_kg(t_c):
    """Latent heat of vaporisation, kJ/kg - Wang et al. (2011), as Module 4a uses."""
    return 2499.5698 - 2.204864 * t_c - 0.001596 * t_c ** 2


def distillate_m3_per_day(heat_kw, gor, t_tbt_c, t_condenser_c):
    """
    MED distillate from a thermal duty: GOR x Q / L, with the latent heat at
    the MEAN CASCADE temperature. (A fixed 2,350 kJ/kg corresponds to about
    63 C, warmer than most cascades that are feasible once the loop pinch is
    enforced.)
    """
    l_kj_kg = latent_heat_kj_kg(0.5 * (t_tbt_c + t_condenser_c))
    return gor * heat_kw / l_kj_kg * 86400.0 / 1000.0
