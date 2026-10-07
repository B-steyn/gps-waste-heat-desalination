"""
Levelised cost of water (LCOW), the model's headline result.

    LCOW = (CAPEX * CRF + OPEX_annual) / water_produced_annual

REAL BASIS. Every cost entering this module is in constant 2026 AUD with
no forward escalation, so the discount rate must also be real. Passing a
nominal rate here would add inflation compensation to cash flows that
contain no inflation and overstate the capital charge by about 27 % at
typical values, moving every LCOW (though not the ranking).

`capital_recovery_factor` therefore takes a real rate and says so; use
`economic_constants.real_rate()` to convert a nominal WACC.
"""

from common.economic_constants import (
    WACC_REAL, ASSET_LIFE_WATER_YEARS, ASSET_LIFE_HEAT_RECOVERY_YEARS,
    OANDM_RATE_ORC_HX, OANDM_RATE_MED_RO, ELECTRICITY_AUD_PER_MWH,
    MEMBRANE_LIFE_YEARS, GPS_AVAILABILITY, DESAL_OWN_AVAILABILITY,
    MEMBRANE_SHARE_OF_RO_OPEX,
)


def capital_recovery_factor(real_discount_rate, life_years):
    """
    CRF = i(1+i)^N / ((1+i)^N - 1)

    `real_discount_rate` must be REAL (see module docstring). At a zero
    rate this degenerates to straight-line 1/N, handled explicitly
    rather than dividing by zero.
    """
    if life_years <= 0:
        raise ValueError("life_years must be positive")
    i = real_discount_rate
    if abs(i) < 1e-12:
        return 1.0 / life_years
    return i * (1.0 + i) ** life_years / ((1.0 + i) ** life_years - 1.0)


def annualise_capex(capex_by_pool_aud, real_discount_rate=WACC_REAL,
                     life_water_years=ASSET_LIFE_WATER_YEARS,
                     life_heat_years=ASSET_LIFE_HEAT_RECOVERY_YEARS):
    """
    Annualised capital charge, with the two CAPEX pools amortised over
    their own lives.

    `capex_by_pool_aud` is a dict with keys 'water' and 'heat_recovery'.
    They are kept separate because they carry different risk: if GPS
    closes, the FGCHE and its heat source stop, but the desalination
    plant and its intake/outfall could in principle be re-powered.
    Amortising both over one life conflates two different exposures.
    """
    water = capex_by_pool_aud.get("water", 0.0)
    heat = capex_by_pool_aud.get("heat_recovery", 0.0)
    return (water * capital_recovery_factor(real_discount_rate, life_water_years) +
            heat * capital_recovery_factor(real_discount_rate, life_heat_years))


def annual_opex_aud(capex_by_pool_aud, net_electricity_kw, operating_hours,
                     membrane_capex_aud=0.0,
                     electricity_aud_per_mwh=ELECTRICITY_AUD_PER_MWH,
                     oandm_heat=OANDM_RATE_ORC_HX, oandm_water=OANDM_RATE_MED_RO,
                     membrane_life_years=MEMBRANE_LIFE_YEARS):
    """
    Annual operating cost.

    `net_electricity_kw` is the SYSTEM net (ORC output less ID fan, loop
    pump and process demand), AVERAGED over `operating_hours`: Module 6
    weights the heat-side terms by the share of hours with flue gas, and
    takes them as their means over the operating records.
    Sign matters and is handled explicitly:
      negative -> the scheme imports power, a cost
      positive -> the scheme exports, a credit
    Both are valued at the same price here. At a generating site the
    buy/sell basis is genuinely ambiguous - imported power could be
    valued at wholesale, at internal transfer cost, or at the
    opportunity cost of forgone export - so a single price is used and
    the ambiguity is swept rather than an arbitrary retail markup being
    asserted.

    Membrane replacement is a COMPONENT OF the fixed O&M rate ("10-20 % of
    RO OPEX"), not an additional charge on top of it: about 15 % of the
    6 % rate, or 0.9 % of RO capital a year. A separate sinking fund of
    20 % of RO capital every 5 years (4 % a year) would count membranes
    twice and overcharge them about 4.4-fold. A useful sanity check: O&M
    at a fixed 6 % of capital cannot exceed the capital charge while 6 %
    is below the CRF (7.36 %).

    Membrane life is a sweep parameter. It scales the membrane
    SUB-COMPONENT of O&M (baseline_life / actual_life): shorter life,
    more frequent replacement, a larger share of O&M.
    """
    water_capex = capex_by_pool_aud.get("water", 0.0)
    heat_capex = capex_by_pool_aud.get("heat_recovery", 0.0)

    # membrane sub-component of the water O&M rate, scaled by life
    life_factor = (MEMBRANE_LIFE_YEARS / membrane_life_years
                   if membrane_life_years > 0 else 1.0)
    membrane_share = MEMBRANE_SHARE_OF_RO_OPEX * life_factor
    non_membrane_share = 1.0 - MEMBRANE_SHARE_OF_RO_OPEX

    membrane =water_capex * oandm_water * membrane_share
    fixed = heat_capex * oandm_heat + water_capex * oandm_water * non_membrane_share

    electricity_mwh = (net_electricity_kw / 1000.0) * operating_hours
    # net export is a credit (negative cost), net import a cost
    electricity_cost = -electricity_mwh * electricity_aud_per_mwh

    return {"fixed_oandm_aud": fixed,
            "electricity_aud": electricity_cost,
            "membrane_aud": membrane,
            "total_aud": fixed + electricity_cost + membrane}


def levelised_cost_of_water(capex_by_pool_aud, water_m3_per_year, net_electricity_kw,
                             operating_hours, membrane_capex_aud=0.0,
                             real_discount_rate=WACC_REAL,
                             electricity_aud_per_mwh=ELECTRICITY_AUD_PER_MWH,
                             life_water_years=ASSET_LIFE_WATER_YEARS,
                             life_heat_years=ASSET_LIFE_HEAT_RECOVERY_YEARS,
                             membrane_life_years=MEMBRANE_LIFE_YEARS):
    """
    LCOW in AUD/m3, with the component breakdown retained so the result
    can be decomposed rather than reported as a single opaque number.

    Returns None for lcow_aud_per_m3 if no water is produced - an
    architecture that produces no water has no levelised cost of water,
    and returning infinity or zero would both be misleading.
    """
    if water_m3_per_year <= 0:
        return {"lcow_aud_per_m3": None, "water_m3_per_year": water_m3_per_year,
                "note": "no water produced - LCOW undefined"}

    annual_capital = annualise_capex(capex_by_pool_aud, real_discount_rate,
                                      life_water_years, life_heat_years)
    opex = annual_opex_aud(capex_by_pool_aud, net_electricity_kw, operating_hours,
                            membrane_capex_aud, electricity_aud_per_mwh,
                            membrane_life_years=membrane_life_years)

    total_annual = annual_capital + opex["total_aud"]

    return {
        "lcow_aud_per_m3": total_annual / water_m3_per_year,
        "capital_component_aud_per_m3": annual_capital / water_m3_per_year,
        "opex_component_aud_per_m3": opex["total_aud"] / water_m3_per_year,
        "electricity_component_aud_per_m3": opex["electricity_aud"] / water_m3_per_year,
        "annual_capital_charge_aud": annual_capital,
        "annual_opex_aud": opex["total_aud"],
        "opex_breakdown": opex,
        "total_capex_aud": sum(capex_by_pool_aud.values()),
        "water_m3_per_year": water_m3_per_year,
        "real_discount_rate": real_discount_rate,
    }


def annual_operating_hours(architecture, gps_availability=GPS_AVAILABILITY,
                            desal_own_availability=DESAL_OWN_AVAILABILITY):
    """
    Annual operating hours, which are NOT the same for every architecture.

    MED is heat-driven: no boiler means no water, so its hours are capped
    by GPS availability (76.2 % measured). RO is electrically driven -
    the waste heat only lowers its SEC - so it can run through boiler
    outages on grid power, capped instead by its own maintenance
    availability.

    Returns hours split into preheated and ambient, because RO's specific
    energy differs between them.
    """
    total_h = 8760.0
    preheated = total_h * gps_availability
    if architecture.upper() in ("MED", "DIRECT_MED", "ORC_MED"):
        return {"total_h": preheated, "preheated_h": preheated, "ambient_h": 0.0}
    if architecture.upper() in ("RO", "ORC_RO", "SWRO"):
        total = total_h * desal_own_availability
        return {"total_h": total, "preheated_h": min(preheated, total),
                "ambient_h": max(total - preheated, 0.0)}
    raise ValueError(f"unknown architecture {architecture!r}")


def blended_sec_kwh_m3(sec_preheated, sec_ambient, hours):
    """
    Output-weighted SEC across preheated and ambient operating hours.
    Water output is proportional to hours at constant plant capacity, so
    weighting by hours is the correct basis.
    """
    total = hours["preheated_h"] + hours["ambient_h"]
    if total <= 0:
        return 0.0
    return (sec_preheated * hours["preheated_h"] + sec_ambient * hours["ambient_h"]) / total
