"""
Composition-weighted flue gas specific heat, from CoolProp ideal-gas species
lookups (Cp0mass; see the note in the function), using the per-record
composition from the coal ultimate analysis and the measured excess air.

A fixed 1.045 kJ/kg-K is close to pure N2 at flue-gas temperatures, a
reasonable first guess since N2 is 70-75 % of the gas by mass, but it misses
the CO2, H2O and O2 content and its variation with temperature and excess
air.

Approximation: SO2/SO3 are ppm-level in the mole balance and contribute
negligibly to the mass-weighted mixture Cp, so they're omitted here rather
than looked up (CoolProp has no direct SO3 fluid model). This is flagged,
not silent.
"""

import CoolProp.CoolProp as CP

_SPECIES = {
    "y_co2": "CarbonDioxide",
    "y_h2o": "Water",
    "y_o2": "Oxygen",
    "y_n2": "Nitrogen",
}

_MOLAR_MASS_KG_PER_MOL = {key: CP.PropsSI("M", fluid) for key, fluid in _SPECIES.items()}


def dynamic_flue_gas_cp(t_celsius, mole_fractions, p_kpa=101.325):
    """
    Mass-weighted mixture Cp (kJ/kg-K) at the given bulk gas temperature,
    from per-species CoolProp Cpmass lookups.

    mole_fractions: dict with keys y_co2, y_h2o, y_o2, y_n2 (as returned by
    common.coal_combustion.combustion_mole_balance). Renormalized to just
    these four species so the SO3/SO2 omission doesn't bias the mass split.
    """
    t_k = t_celsius + 273.15
    p_pa = p_kpa * 1000.0

    mole_total = sum(mole_fractions[key] for key in _SPECIES)
    mass_fractions = {}
    mass_total = 0.0
    for key in _SPECIES:
        mass = (mole_fractions[key] / mole_total) * _MOLAR_MASS_KG_PER_MOL[key]
        mass_fractions[key] = mass
        mass_total += mass

    cp_mix_j_per_kgk = 0.0
    for key, fluid in _SPECIES.items():
        w_i = mass_fractions[key] / mass_total
        # IDEAL-GAS specific heat (Cp0mass), not a pure-fluid lookup at the
        # mixture's total pressure. Pure water at 1 atm is liquid below 100 C,
        # so a pure-fluid lookup would return about 4.2 kJ/kg-K for the vapour
        # fraction instead of about 1.89. Flue-gas water vapour sits at a
        # partial pressure of about 0.08 atm, well above its own dew point,
        # and behaves as an ideal gas, as do N2, O2 and CO2 here.
        cp_i = CP.PropsSI("Cp0mass", "T", t_k, "P", p_pa, fluid)
        cp_mix_j_per_kgk += w_i * cp_i

    return cp_mix_j_per_kgk / 1000.0  # kJ/kg-K
