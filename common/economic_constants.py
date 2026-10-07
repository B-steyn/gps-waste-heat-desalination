"""
Economic and financial parameters for the levelised-cost model (Module 6).

Kept separate from common/constants.py (physical and site constants)
because cost constants fail differently: a wrong physical constant usually
gives an obviously wrong temperature, whereas a wrong cost constant gives a
plausible-looking number that is simply incorrect. Every entry therefore
carries a status tag:

  VERIFIED   - checked against a primary source
  PARTLY     - method or concept confirmed, exact value not
  UNVERIFIED - plausible and retained, not independently checked
  ASSUMPTION - a stated modelling choice, not a literature value

All costs are constant 2026 Australian dollars (a real basis).
"""

# ==========================================
# currency, escalation and time basis
# ==========================================
COST_BASE_YEAR = 2026
FX_USD_TO_AUD = 1.43          # VERIFIED - July 2026 (~0.699 USD/AUD)

# Chemical Engineering Plant Cost Index. Each correlation is escalated from
# its own base year.
CEPCI = {
    2019: 607.5,    # PARTLY     - Bhojwani et al. base, approximate annual average
    2020: 596.2,    # UNVERIFIED - Shamoushaki et al. (QUE$TOR database year)
    2026: 815.0,    # VERIFIED   - about 800 since 2024, modest rise into 2026
}

# ==========================================
# discount basis - REAL, not nominal
# ==========================================
# Every cost is in constant 2026 dollars with no forward escalation, so the
# cash flows are real and must be discounted at a real rate. Discounting them
# at the nominal rate would add inflation compensation to numbers that contain
# no inflation and overstate the capital charge by about 27 % at these values.
WACC_NOMINAL = 0.08           # PARTLY - the Queensland Competition Authority's water
                              # determination for Gladstone Area Water Board is 7.39 %
INFLATION = 0.025             # ASSUMPTION - midpoint of the RBA 2-3 % target band


def real_rate(nominal=WACC_NOMINAL, inflation=INFLATION):
    """Fisher relation: r_real = (1 + r_nominal) / (1 + inflation) - 1."""
    return (1.0 + nominal) / (1.0 + inflation) - 1.0


WACC_REAL = real_rate()       # 5.37 % from 8 % nominal at 2.5 % inflation

# ==========================================
# asset life
# ==========================================
# Queensland's 2050 net-zero target is 24 years away and station closure
# dates are not public, while a water-scarcity case would tend to extend the
# life of water infrastructure beyond the generating asset. The two capital
# pools are kept separate: if the station closes, the flue-gas exchanger and
# its heat stop, but the desalination plant and its intake and outfall could
# in principle be re-powered.
ASSET_LIFE_WATER_YEARS = 25.0          # ASSUMPTION - desalination plant, intake, outfall
ASSET_LIFE_HEAT_RECOVERY_YEARS = 25.0  # ASSUMPTION - exchangers, loop, ORC; carries station-life risk

# ==========================================
# installed-cost build-up (purchased equipment -> installed)
# ==========================================
LANG_FACTOR = 3.50            # PARTLY - 3.0-5.0 range; lean against the 4.7-6.0 typical of
                              # fluid-processing plant, reflecting a retrofit
LOCATION_FACTOR_AU = 1.21     # VERIFIED - Towler & Sinnott, Chemical Engineering Design,
                              # 3rd ed., p. 265: Australia relative to the US Gulf Coast.
                              # A location factor converts an INSTALLED cost between
                              # locations, so it applies to the exchangers' installed cost
                              # and to the (installed, largely US-based) desalination
                              # capital of Bhojwani et al.
CONTINGENCY = 0.15            # PARTLY - 10-20 %; consistent with AACE 16R-90
EPC_INDIRECT = 0.20           # PARTLY - 15-30 %; consistent with NETL/AACE guidance

# ==========================================
# heat exchanger costing
# ==========================================
# Shamoushaki et al. (2021), Energies 14, 2665, fitted to a 2020 QUE$TOR
# database:  C = log(A) + a A^2 + b A + c  [USD], valid for 1-3500 m2, the
# only published range that covers this model's 1,300-3,300 m2 exchangers.
# The quadratic term is negative, so the fit peaks near 7,405 m2 and falls
# beyond it; the validity bound is enforced in costing.hx_purchased_cost_aud.
SHAMOUSHAKI_SHELL_TUBE = {"a": -0.06395, "b": 947.2, "c": 227.9,
                           "a_min_m2": 1.0, "a_max_m2": 3500.0, "base_year": 2020}

# Material multipliers relative to carbon steel (SOW reference [14]). The
# Shamoushaki basis already has stainless tubes, so 316L takes no uplift and
# 904L is priced relative to 316 (costing._material_factor).
HX_MATERIAL_MULTIPLIER_316 = (1.3, 1.8)   # VERIFIED - bare module, carbon steel -> 304/316
HX_MATERIAL_MULTIPLIER_904L = (4.0, 5.0)  # ASSUMPTION - no published 904L factor exists

# ==========================================
# ORC costing
# ==========================================
ORC_SPECIFIC_COST_AUD_PER_KW = (2000.0, 5000.0)  # VERIFIED - SOW reference [45], sub-200 C systems
ORC_SPECIFIC_COST_BASELINE = 3500.0              # ASSUMPTION - midpoint

# ==========================================
# desalination capital - Bhojwani et al. (2019)
# ==========================================
# Bhojwani, S., Topolski, K., Mukherjee, R., Sengupta, D. & El-Halwagi, M.M.
# (2019). Technology review and data analysis for cost assessment of water
# treatment systems. Science of the Total Environment 651, 2749-2761.
#
# One paper costs MED and SWRO with one method, which is what a comparison
# between the two needs. Its Table 5 gives the capital contribution to unit
# product cost (USD/m3):
#   capacity m3/d :   3785   18925   37850   189250
#   MED (PR = 12) :  0.381   0.275   0.219    0.125
#   SWRO          :  0.305   0.220   0.175    0.100
# so MED/SWRO = 1.25 at every capacity, and both scale with the same
# exponent: -0.2864 on unit-product cost (a total-CAPEX exponent of 0.714,
# R2 = 0.968).
BHOJWANI_MED_OVER_SWRO = 1.25          # VERIFIED - Table 5, constant across capacity
BHOJWANI_SCALE_EXPONENT = -0.2864      # VERIFIED - unit-product basis
BHOJWANI_SWRO_UNIT_COEFF_USD = 3.4325  # VERIFIED - fitted, unit capital = k Q^e (USD/m3)
BHOJWANI_BASE_YEAR = 2019

# How Table 5 was built, so that it can be reversed into a capital cost. The
# paper's Section 4.3: "The unit product cost is the sum of the capital cost
# depreciated over the plant life and operating cost per m3 water treated.
# Depreciation is usually taken as 20 yrs without salvage value", with plant
# availability "usually taken as 90-95 %". The capital part of unit product
# cost is therefore
#     capital / 20 / (capacity x 365 x availability)      (no interest)
# and capital per m3/day is Table 5 x 365 x availability x 20. That capital is
# then annualised at this study's own discount rate, like every other item.
#
# Caveat: Table 5 cannot be reproduced from the paper's own Fig. 4 capital
# correlations (SWRO $M = 0.0046915 Q^0.87547, MED $M = 0.0014965 Q^0.9988)
# with any single annualisation, so Table 5 rests on data the paper does not
# show. At this study's capacity, Fig. 4 used directly gives about 13 % less
# SWRO capital, inside the paper's own stated error of +/-20-30 %.
BHOJWANI_DEPRECIATION_YEARS = 20.0     # VERIFIED - Section 4.3: straight line, no salvage, no interest
BHOJWANI_PLANT_AVAILABILITY = 0.95     # VERIFIED range 0.90-0.95 (Section 4.3); 0.95 matches this
                                       # model's own desalination-plant availability
DESAL_COST_ANCHOR = "bhojwani_global"  # the cost basis: Bhojwani et al., located to Australia

# Performance-ratio correction for MED. Table 5's MED figures are quoted at
# PR = 12, and the paper states that capital rises with PR because a higher
# PR needs more effects: "The increase in per capita capital cost from the
# base case of PR = 8 is taken as 10% for systems with PR = 10 and 25% for
# PR = 12." The PR = 12 figure is divided back to the PR = 8 base, and
# costing.med_capex_gor_factor applies the uplift for the GOR actually run.
MED_PR12_TO_PR8_CORRECTION = 1.25    # VERIFIED - the paper's own uplift

# MED commercial-availability penalty. Australia has no MED industry (no
# local supply chain, commissioning or O&M experience), while SWRO is built
# routinely. No published figure exists, so it is a declared assumption;
# assuming no penalty would itself be an unstated assumption favouring MED.
MED_COMMERCIAL_AVAILABILITY_PENALTY = 1.20   # ASSUMPTION - first of a kind in Australia

# ==========================================
# operating costs
# ==========================================
OANDM_RATE_ORC_HX = 0.04      # UNVERIFIED - Ruiz-Garcia & Nuez (2017), 3-6 % band
OANDM_RATE_MED_RO = 0.06      # UNVERIFIED - Ruiz-Garcia & Nuez (2017), 5-8 % band

# Electricity, valued at one price for imports and exports. At a generating
# site the basis is genuinely ambiguous (wholesale, internal transfer cost or
# forgone export), so a single wholesale price is used and swept.
ELECTRICITY_AUD_PER_MWH = 65.00                # VERIFIED - AEMO Queensland averages, 2025-26 and 2026-27
ELECTRICITY_SWEEP_AUD_PER_MWH = (58.0, 128.0)  # VERIFIED - AEMO QED Q4-2025 low; AER CY2024 Queensland average

# Membranes are a component of the RO O&M rate ("10-20 % of RO OPEX"), not a
# separate charge on top of it; membrane life scales that component
# (lcow.annual_opex_aud).
MEMBRANE_LIFE_YEARS = 5.0                  # UNVERIFIED - Voutchkov (2018)
MEMBRANE_SHARE_OF_RO_OPEX = 0.15           # PARTLY - midpoint of the 10-20 % band

# ==========================================
# availability - and why it differs by architecture
# ==========================================
# 461 of the 605 six-hourly records (2026-01-05 to 2026-06-05) have a unit
# load of at least 1 MW. The cleaned dataset holds only those, so plant
# downtime is invisible to the thermodynamic model but matters to the
# economics, and not equally:
#   MED is heat-driven: no boiler, no heat, no water. Its hours are capped by
#       the station's availability.
#   RO is electrically driven: the heat only lowers its specific energy, so it
#       runs through boiler outages on grid power, at a higher specific energy.
GPS_AVAILABILITY = 0.7620      # VERIFIED - 461/605 records with load >= 1 MW

# The desalination plant's own availability (maintenance, membrane cleaning),
# independent of the boiler. It limits RO; MED is limited by the boiler.
DESAL_OWN_AVAILABILITY = 0.95  # ASSUMPTION - typical for a continuously run membrane plant
