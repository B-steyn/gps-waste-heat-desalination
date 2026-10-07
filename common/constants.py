"""
Physical, chemical and site constants for the Gladstone Power Station (GPS)
thermo-economic model. One source of truth, imported by every module.
"""

import CoolProp.CoolProp as _CP

# ==========================================
# coal ultimate/proximate analysis (GPS typical coal analysis)
# ==========================================
C_DAF = 0.804
H_DAF = 0.051
N_DAF = 0.021
O_DAF = 0.116
S_DAF = 0.008

MOIST_AD = 9.5      # % air-dried moisture
ASH_AD = 7.5         # % air-dried ash
TOTAL_MOIST_AR = 0.17  # as-received total moisture fraction

# ==========================================
# combustion / acid dew point chemistry
# ==========================================
CONVERSION_SO2_TO_SO3 = 0.010  # furnace conversion rate of SO2 to SO3

# The single highest-leverage uncited input to the acid dew point; to be
# confirmed with the station rather than swept. UNCITED.
ASH_CAPTURE_EFF = 0.90  # SO3 captured by the alkaline ash cake in the fabric filter

# ==========================================
# ambient / dead-state reference
# ==========================================
P_AMBIENT_KPA = 101.325
T_AMBIENT_CELSIUS = 25.0  # dead-state reference temperature (°C)

# ==========================================
# tap point
# ==========================================
# The flue-gas exchanger sits in the duct downstream of the fabric filter, not
# at the economiser exit. Measured gas there runs about 93-125 C (mean about
# 115 C), against about 285-320 C at the air-heater inlet. The site acid dew
# point (calculate_dynamic_adp, on the real coal assay) is about 105 C, so the
# exchanger runs below it at typical and low loads. That is accepted as a
# design condition: post-filter gas carries little particulate, which reduces
# (but does not remove) the ash-acid fouling risk.
TAP_POINT_NOTE = (
    "FGCHE tap point is post-fabric-filter duct (~93-125 C measured), not "
    "economiser exit (~285-320 C at the air-heater inlet). "
    "Site ADP ~105 C; FGCHE runs below ADP at typical/low loads by design, "
    "accepted because post-FF gas has low particulate loading."
)

# ==========================================
# heat exchanger / intermediate loop geometry and materials
# ==========================================
P_WATER_LOOP_PA = 1_000_000.0  # pressurised closed loop water pressure (1.0 MPa / 10 bar)
LOSS_BUFFER_DELTA_T = 5.0  # loop supply sits this far below the extraction track (°C)
HX_SHELL_EFFICIENCY = 0.985  # 1.5 % of the recovered heat lost from the loop to surroundings

PIPE_LENGTH_M = 300.0
PUMP_EFFICIENCY = 0.75
PIPE_FRICTION_PAM = 150.0
RHO_LIQUID_WATER = 997.0

# Duct cross-section, measured at an expansion joint just downstream of the
# fabric filter, i.e. the ID fan suction duct (one of the two per unit).
# 24.52 m2.
DUCT_WIDTH_M = 5.79
DUCT_HEIGHT_M = 4.235

# The gas path. Each unit's boiler has a bilateral gas path: the dataset
# carries A-side and B-side columns for the air heater, fabric filter (inlet
# and outlet temperature and pressure), dust burden, ID fan current and ID fan
# vane position. Both fans run in all 461 records, balanced (A/B current ratio
# 1.017). So the unit has two parallel trains of about 178 kg/s each, and the
# measured duct is on the fan suction side:
#
#   boiler -+- A: air heater -> fabric filter -=[TAP]=- ID fan A -+
#           +- B: air heater -> fabric filter -=[TAP]=- ID fan B -+-> merged
#                                                                     duct -> stack
#
# pre_fan  : two ducts of about 178 kg/s, each with its own fan; face velocity
#            about 8.6 m/s. The measured location, and the one modelled.
# post_fan : one merged duct carrying the whole unit (about 356 kg/s), both
#            fans sharing the pressure drop; face velocity about 17 m/s, and
#            since dP goes as v^2 the fan penalty would be about 5.6x worse.
#
# The 2,450 kW nameplate (Siemens 1RQ 1802-8Js70-Z, 6.6 kV) is PER FAN: at
# 6.6 kV, PF 0.88 and 96 % efficiency a 2,450 kW motor draws 253.7 A, and the
# logged maximum of 226.8 A is 89.4 % of that. Read as a rating for the pair it
# would imply 178.8 % of rated current.
TAP_LOCATION = "pre_fan"

if TAP_LOCATION == "pre_fan":
    N_GAS_TRAINS = 2       # parallel ducts at the tap
    N_FANS_PER_DUCT = 1    # each duct has its own ID fan
else:
    N_GAS_TRAINS = 1       # merged duct carries the whole unit
    N_FANS_PER_DUCT = 2    # both fans push through it

# How many of the parallel ducts receive an exchanger: a design choice, unlike
# N_GAS_TRAINS. Both are fitted, so the exchanger takes the unit's full gas
# flow; each fan still sees only its own duct's pressure drop.
N_TRAINS_FITTED = 2
ID_FAN_RATING_KW = 2450.0
ID_FAN_VOLTAGE_V = 6600.0
ID_FAN_MOTOR_EFFICIENCY = 0.96
TUBE_DO_M = 0.0254
TUBE_WALL_M = 0.0020
PITCH_RATIO = 1.75

K_316L = 16.3  # W/m-K
K_904L = 13.0  # W/m-K

# flue gas transport properties (for Nusselt and Reynolds numbers; the heat
# capacity is composition-dependent, in common/gas_properties.py)
K_GAS_FLUE = 0.032
MU_GAS_FLUE = 0.000022
PRANDTL_GAS = 0.71

FOULING_RESISTANCE_BASELINE = 0.0002   # m2K/W at the reference dust burden - UNCITED
FOULING_DUST_NORMALIZER = 11.0         # reference dust burden, mg/m3 - UNCITED

TUBE_ROW_DP_FACTOR = 0.25   # pressure drop per tube row in velocity heads (an Euler number) - UNCITED

FAN_EFFICIENCY = 0.78       # induced-draft fan efficiency, held at every load - UNCITED

# Zukauskas (1972) tube-bank correlation coefficients: Nu = C * Re^m * Pr^n
ZUKAUSKAS_C = 0.35
ZUKAUSKAS_M = 0.6
ZUKAUSKAS_N = 0.36

# Default loop return temperature by architecture (°C), used where no
# optimised design exists for a track (Module 3's fallback rating).
ARCHITECTURE_WATER_RETURN_C = {
    "direct_med": 60.0,
    "orc_med": 50.0,
    "orc_ro": 35.0,
}

# flue-gas extraction tracks: the exchanger's gas outlet temperature (°C)
TARGET_TEMPERATURES = [100, 95, 90, 85]

# ==========================================
# Module 5: ORC cycle parameters
# ==========================================
# Default fluid for standalone calls. The reported designs use R1234ze(E),
# passed explicitly (module5_orc.path_b_transcritical.ORC_TRANS_FLUID).
ORC_WORKING_FLUID = "R245fa"
ORC_TURBINE_ISENTROPIC_EFF = 0.80  # Li, J. et al.
ORC_PUMP_ISENTROPIC_EFF = 0.75     # Zhu et al. and Li, J. et al. agree
ORC_GENERATOR_EFF = 0.96           # Zhu et al., corroborated by Sun et al. and Auld et al.

# Net output = generator efficiency x turbine work - pump work - auxiliaries.
# The auxiliaries (cooling system, controls, lube oil pumps; separate from the
# working-fluid pump) are taken as 3 % of turbine work, a common figure for
# small and medium ORCs. UNCITED.
ORC_PARASITIC_FRACTION_OF_GROSS = 0.03

# Evaporator pinch. Zhu et al. use 10 K and Li, J. et al. 5 K; ORC exchanger
# design has tightened since those papers. Net output rises monotonically with
# evaporating temperature (no interior optimum), so the tighter pinch is
# strictly better and costs MED nothing in heat supply.
ORC_EVAPORATOR_PINCH_C = 5.0
ORC_CONDENSER_PINCH_C = 5.0        # both papers agree; also the ORC+RO seawater approach
ORC_SUPERHEAT_C = 5.0              # Li, J. et al.

# ORC+MED: the condenser must stay compatible with MED's top brine temperature.
ORC_MED_TC_RANGE_C = (65.0, 75.0)
# The ORC+MED condensing temperature, scanned across its range in 5 K steps.
# Tc sets the MED top brine temperature (Tc - pinch), which caps the number of
# effects, so Tc is a design variable chosen by lowest LCOW.
ORC_MED_TC_CANDIDATES_C = (65.0, 70.0, 75.0)

# ==========================================
# Module 6: RO thermal enhancement
# ==========================================
# Membrane temperature ceiling. Commercial membranes are typically limited to
# 35-40 C, and Yagnambhatt et al. model up to 50 C to cover recent
# high-temperature materials. Carried as two explicit design cases.
RO_CEILING_CURRENT_MEMBRANE_C = 38.0   # current membranes; within Yagnambhatt's "typical" band
RO_CEILING_ADVANCED_MEMBRANE_C = 50.0  # Yagnambhatt's upper tested bound (see also Chen et al.)

# ORC+RO condensing temperature: ceiling + pinch. The lowest Tc that gives the
# full preheat at a given ceiling: hotter only costs electricity for no extra
# benefit (the feed is capped at the ceiling), cooler leaves benefit unused.
ORC_RO_TC_CURRENT_MEMBRANE_C = RO_CEILING_CURRENT_MEMBRANE_C + ORC_CONDENSER_PINCH_C
ORC_RO_TC_ADVANCED_MEMBRANE_C = RO_CEILING_ADVANCED_MEMBRANE_C + ORC_CONDENSER_PINCH_C

# Seawater specific heat capacity (kJ/kg-K) at 35 g/kg, from the MIT seawater
# correlations (Sharqawy, Lienhard & Zubair 2010, Desalination and Water
# Treatment 16, 354-380) as implemented in CoolProp's INCOMP::MITSW. Evaluated
# at 32 C, the middle of the RO feed's preheat (25.9 -> 38 C); it varies by
# only 0.1 % over that span.
SEAWATER_CP_KJ_KGK = _CP.PropsSI("C", "T", 32.0 + 273.15, "P", 101325.0, "INCOMP::MITSW[0.035]") / 1000.0

# RO permeate recovery. One definition: every specific-energy call passes it,
# so water output and the RO transport model use the same recovery.
RO_RECOVERY = 0.45
