"""
Equipment costing.

Every correlation is escalated from its own base year, so `escalate()`
takes the source year explicitly and no function carries a default for it.

All functions return AUD in COST_BASE_YEAR unless the name says otherwise.
Purchased-equipment cost and installed cost are kept distinct:
`installed_cost()` applies the Lang, location, contingency and EPC build-up,
and callers say which they want.
"""

import math

from common.economic_constants import (
    CEPCI, COST_BASE_YEAR, FX_USD_TO_AUD, LANG_FACTOR, LOCATION_FACTOR_AU,
    CONTINGENCY, EPC_INDIRECT, SHAMOUSHAKI_SHELL_TUBE,
    HX_MATERIAL_MULTIPLIER_316, HX_MATERIAL_MULTIPLIER_904L,
    ORC_SPECIFIC_COST_BASELINE,
    BHOJWANI_MED_OVER_SWRO, BHOJWANI_SCALE_EXPONENT, BHOJWANI_SWRO_UNIT_COEFF_USD,
    BHOJWANI_BASE_YEAR, DESAL_COST_ANCHOR,
    BHOJWANI_DEPRECIATION_YEARS, BHOJWANI_PLANT_AVAILABILITY,
    MED_COMMERCIAL_AVAILABILITY_PENALTY, MED_PR12_TO_PR8_CORRECTION,
)


def escalate(cost, from_year, to_year=COST_BASE_YEAR):
    """CEPCI escalation. from_year must be the correlation's own base year."""
    if from_year not in CEPCI:
        raise ValueError(f"no CEPCI value for {from_year}; known years: {sorted(CEPCI)}")
    if to_year not in CEPCI:
        raise ValueError(f"no CEPCI value for {to_year}; known years: {sorted(CEPCI)}")
    return cost * (CEPCI[to_year] / CEPCI[from_year])


def usd_to_aud(cost_usd):
    return cost_usd * FX_USD_TO_AUD


# 904L uplift relative to the correlation's stainless basis, derived rather
# than cited: 904L is 4-5x carbon steel and 316 is 1.3-1.8x, so 904L over 316
# is (4-5)/(1.3-1.8). No published table carries a 904L factor directly.
_904L_OVER_316 = (HX_MATERIAL_MULTIPLIER_904L[0] / HX_MATERIAL_MULTIPLIER_316[1],
                   HX_MATERIAL_MULTIPLIER_904L[1] / HX_MATERIAL_MULTIPLIER_316[0])


def _material_factor(material):
    """
    Material factor relative to the Shamoushaki correlation's own basis, which
    is a carbon steel shell with STAINLESS tubes, not carbon steel throughout.

    Applying a carbon-steel-to-316 multiplier (1.3-1.8x) on top of this
    correlation would count the tube material twice. A cross-check against
    Corripio at 1,000 m2 confirms the basis: the correlation sits at 4.44x
    Corripio's carbon steel (implausible for a carbon-steel unit) but 1.47x
    Corripio's 316L (plausible for two independent correlations).

    So 316L takes no uplift: in a shell-and-tube unit the tubes carry the
    whole wetted surface and most of the material cost, and the stainless-tube
    basis already represents 316L service for this duty.
    """
    if material in ("316L", "316", "SS316", "base"):
        return 1.0
    if material in ("904L", "904"):
        lo, hi = _904L_OVER_316
        return (lo + hi) / 2.0
    if material in ("CS", "carbon_steel"):
        raise ValueError(
            "this correlation's basis includes stainless tubes, so it cannot give a "
            "carbon-steel cost. Use a carbon-steel correlation such as Corripio's.")
    raise ValueError(f"unknown material {material!r}")


def hx_purchased_cost_aud(area_m2, material="316L", clamp_to_validity=True):
    """
    Shell-and-tube purchased equipment cost, Shamoushaki et al. (2021).

        C = log(A) + a*A^2 + b*A + c        [1-3500 m2]

    The validity bound is enforced: the quadratic coefficient is negative, so
    the fit peaks near 7,405 m2 and declines beyond it, and an unguarded call
    at 8,000 m2 would return less than at 3,500 m2. Above the bound the duty
    is split across identical shells, which is also what would be built.
    """
    p = SHAMOUSHAKI_SHELL_TUBE
    if area_m2 <= 0:
        return 0.0
    if area_m2 > p["a_max_m2"]:
        if not clamp_to_validity:
            raise ValueError(
                f"area {area_m2:.0f} m2 exceeds the correlation's {p['a_max_m2']:.0f} m2 "
                f"validity bound, beyond which the fit is non-physical (declining). "
                f"Split the duty across shells, or pass clamp_to_validity=False only if "
                f"you have separately justified the extrapolation.")
        n_shells = math.ceil(area_m2 / p["a_max_m2"])
        return n_shells * hx_purchased_cost_aud(area_m2 / n_shells, material=material)

    a, b, c = p["a"], p["b"], p["c"]
    cost_usd = math.log(area_m2) + a * area_m2 ** 2 + b * area_m2 + c
    cost_aud = usd_to_aud(escalate(cost_usd, p["base_year"]))
    return cost_aud * _material_factor(material)


def orc_cost_aud(w_net_kw, specific_cost_aud_per_kw=ORC_SPECIFIC_COST_BASELINE):
    """
    ORC package cost from its rated output, at A$2,000-5,000 per kW for
    sub-200 C systems (SOW reference [45]).

    `w_net_kw` is the cycle's net output at the design record: after the ORC's
    own pump, generator and auxiliary load, but before the ID fan and loop
    pump. The machine is specified by what it generates; the fan is a cost of
    the flue-gas exchanger, which every heat-recovery architecture pays.
    """
    return max(w_net_kw, 0.0) * specific_cost_aud_per_kw


def desal_specific_capex_aud_per_m3_day(capacity_m3_per_day, technology="SWRO",
                                          anchor=DESAL_COST_ANCHOR,
                                          med_availability_penalty=MED_COMMERCIAL_AVAILABILITY_PENALTY):
    """
    Desalination capital in AUD per (m3/day), on one basis for both
    technologies: Bhojwani et al. (2019), Table 5, which costs MED and SWRO
    with one method in one paper. MED is 1.25x SWRO at matched capacity and
    both scale with the same exponent (-0.2864 on unit-product cost).

    Table 5 is the capital contribution to unit product cost (USD per m3). The
    paper's Section 4.3 states how it was built: capital depreciated straight
    line over 20 years, no salvage and no interest, over a plant available
    90-95 % of the year. Reversing it by the same method recovers the paper's
    capital; that capital is then escalated, converted to AUD, moved to
    Australia with the location factor (it is an installed, largely US-based
    cost) and annualised at this study's own discount rate in lcow.py, like
    every other capital item.

    MED is given no separate correlation: it is the SWRO figure times
    Bhojwani's 1.25, so that the two technologies stay on one basis.
    """
    if capacity_m3_per_day <= 0:
        return 0.0

    if anchor != "bhojwani_global":
        raise ValueError(f"unknown cost basis {anchor!r}; use 'bhojwani_global'")
    unit_usd = BHOJWANI_SWRO_UNIT_COEFF_USD * capacity_m3_per_day ** BHOJWANI_SCALE_EXPONENT
    unit_aud = usd_to_aud(escalate(unit_usd, BHOJWANI_BASE_YEAR))
    swro_specific = (unit_aud * 365.0 * BHOJWANI_PLANT_AVAILABILITY * BHOJWANI_DEPRECIATION_YEARS
                     * LOCATION_FACTOR_AU)

    if technology.upper() in ("SWRO", "RO"):
        return swro_specific
    if technology.upper() == "MED":
        # Bhojwani's 1.25x ratio is quoted at PR = 12, which the paper says
        # carries a 25 % capital uplift over its PR = 8 base. It is divided
        # back to PR = 8 here, and med_capex_gor_factor() adds the uplift for
        # the GOR actually run. The commercial-availability penalty is a
        # declared assumption: Australia has no MED industry.
        return (swro_specific * BHOJWANI_MED_OVER_SWRO
                / MED_PR12_TO_PR8_CORRECTION
                * med_availability_penalty)
    raise ValueError(f"unknown technology {technology!r}")


def med_capex_gor_factor(gor):
    """
    Performance-ratio correction to MED capital, relative to the PR = 8 base
    built into desal_specific_capex_aud_per_m3_day().

    Bhojwani quotes MED capital at PR = 12, with a stated 25 % uplift over
    PR = 8:
        GOR <= 8   -> 1.0   (the base; the paper states no relationship below)
        8 .. 12    -> linear to 1.25
        GOR >= 12  -> 1.25

    LIMITATION: no effect-count dependence below GOR 8. A low-GOR plant pays
    the same specific capital as a GOR-8 plant, which overstates it, since
    fewer effects need less evaporator surface. This errs against MED.
    """
    pr_base, pr_top = 8.0, 12.0
    if gor <= pr_base:
        return 1.0
    if gor >= pr_top:
        return MED_PR12_TO_PR8_CORRECTION
    return 1.0 + (gor - pr_base) / (pr_top - pr_base) * (MED_PR12_TO_PR8_CORRECTION - 1.0)


def med_cost_aud(capacity_m3_per_day, anchor=DESAL_COST_ANCHOR,
                  med_availability_penalty=MED_COMMERCIAL_AVAILABILITY_PENALTY):
    """MED total capital, on the same basis as SWRO."""
    return capacity_m3_per_day * desal_specific_capex_aud_per_m3_day(
        capacity_m3_per_day, "MED", anchor, med_availability_penalty)


def swro_cost_aud(capacity_m3_per_day, anchor=DESAL_COST_ANCHOR):
    """SWRO total capital, on the same basis as MED."""
    return capacity_m3_per_day * desal_specific_capex_aud_per_m3_day(
        capacity_m3_per_day, "SWRO", anchor)


def installed_cost(purchased_equipment_cost_aud, lang=LANG_FACTOR,
                    location=LOCATION_FACTOR_AU, contingency=CONTINGENCY,
                    epc=EPC_INDIRECT):
    """
    Purchased equipment cost -> total installed cost.

    The Lang factor converts purchased cost to direct installed cost, the
    location factor moves it to Australia, and contingency and EPC indirects
    are applied to the resulting direct cost. The Lang factor (3.5) is lean
    against the 4.7-6.0 typical of fluid-processing plant; it is a retrofit
    judgement and is swept.
    """
    direct = purchased_equipment_cost_aud * lang * location
    return direct * (1.0 + contingency + epc)


# ---------------------------------------------------------------------------
# ORC exchanger costing
# ---------------------------------------------------------------------------
# Pricing the ORC on A$/kW of output alone would miss most of its cost here:
# the designs run at well under 1 % thermal efficiency, far below the 8-11 %
# of the literature the A$/kW figures come from, yet each still contains two
# exchangers carrying about 12 MW. They are priced with the same Shamoushaki
# correlation as the flue-gas exchanger. The working fluid affects material
# and U, not the A$/m2 basis, and U is uncertain, so it is carried as a band.
ORC_U_BAND_W_M2K = (500.0, 1000.0, 2000.0)


def _lmtd(dt1, dt2):
    """Counterflow LMTD. Returns None if either approach is non-positive."""
    if dt1 <= 0.0 or dt2 <= 0.0:
        return None
    if abs(dt1 - dt2) < 1e-9:
        return dt1
    return (dt1 - dt2) / math.log(dt1 / dt2)


def area_from_pinch_profile_m2(profile, q_total_w, u_w_m2k):
    """
    Exchanger area integrated slice by slice along a discretised T-Q profile
    from common/pinch_check.py:

        A = sum( dQ / (U * dT_lm,slice) )

    A single overall LMTD cannot represent the ORC evaporator: the loop water
    is below the working fluid's saturation temperature over part of the
    exchanger (it is preheating liquid there, not boiling it), and a
    single-zone LMTD would return a negative approach. Integrating the
    composite curves handles the preheat zone and the boiling plateau in one
    pass, on the same discretisation the pinch check uses.

    Each slice carries its own share of the duty, from its end points'
    `duty_fraction` (the pinch check adds the bubble and dew points to an
    evenly spaced grid, so slices are not all equal). A non-positive approach
    anywhere returns None, since that slice would need unbounded area.
    `q_total_w` is in watts.
    """
    n = len(profile) - 1
    if n < 1 or q_total_w <= 0:
        return None
    area = 0.0
    for i in range(n):
        d1 = profile[i]["approach_k"]
        d2 = profile[i + 1]["approach_k"]
        if d1 <= 0.0 or d2 <= 0.0:
            return None
        f0 = profile[i].get("duty_fraction", i / n)
        f1 = profile[i + 1].get("duty_fraction", (i + 1) / n)
        dq_w = (f1 - f0) * q_total_w
        dt_lm = d1 if abs(d1 - d2) < 1e-9 else (d1 - d2) / math.log(d1 / d2)
        area += dq_w / (u_w_m2k * dt_lm)
    return area


def orc_exchanger_cost_aud(q_evap_kw, q_cond_kw, evap_profile,
                            t_cond_c, t_sink_in_c, t_sink_out_c,
                            u_w_m2k=1000.0, material="316L",
                            precomputed_evap_area_m2=None):
    """
    Evaporator and condenser area and installed cost for one ORC.

    Evaporator: integrated over its composite curves (see above).
    Condenser: single-zone LMTD against the sink. Condensation dominates the
    duty, but the desuperheating zone is omitted, so this errs low on area.
    """
    # The optimiser stores the evaporator's UA (integrated over the composite
    # curves), so callers can price it at any U as area = UA/U without
    # re-solving the cycle.
    if precomputed_evap_area_m2 is not None:
        area_evap = precomputed_evap_area_m2
    else:
        area_evap = area_from_pinch_profile_m2(evap_profile, q_evap_kw * 1000.0, u_w_m2k)

    d1, d2 = t_cond_c - t_sink_out_c, t_cond_c - t_sink_in_c
    lmtd_cond = _lmtd(d1, d2)
    area_cond = (q_cond_kw * 1000.0 / (u_w_m2k * lmtd_cond)) if lmtd_cond else None

    cost_evap = hx_purchased_cost_aud(area_evap, material=material) if area_evap else 0.0
    cost_cond = hx_purchased_cost_aud(area_cond, material=material) if area_cond else 0.0

    return {
        "u_w_m2k": u_w_m2k, "lmtd_cond_k": lmtd_cond,
        "area_evap_m2": area_evap, "area_cond_m2": area_cond,
        "purchased_evap_aud": cost_evap, "purchased_cond_aud": cost_cond,
        "installed_aud": installed_cost(cost_evap + cost_cond),
    }


def fgche_cost_aud(area_total_m2, n_trains, material="316L"):
    """
    Installed cost of the flue-gas exchanger, as n_trains SEPARATE units.

    The unit has two parallel gas trains, each with its own ID fan and inlet
    duct, so a two-train installation is two physical exchangers, not one of
    twice the area. The Shamoushaki fit is non-linear, so 2 C(A) != C(2A):
    costing per train is both the physical case and the conservative one.
    """
    if n_trains < 1:
        raise ValueError("n_trains must be >= 1")
    per_train = area_total_m2 / n_trains
    return n_trains * installed_cost(hx_purchased_cost_aud(per_train, material=material))
